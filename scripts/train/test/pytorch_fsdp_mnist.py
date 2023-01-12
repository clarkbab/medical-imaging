import argparse
import functools
import os
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialise process group - this waits for every process in the group (of size 'world_size')
    # to make this call before proceeding.
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print(f'Hello from process {dist.get_rank()}/{dist.get_world_size()}')

def teardown_process():
    dist.destroy_process_group()    # Destroy process group.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  # Track total loss and number of samples processed.
    sampler.set_epoch(epoch)    # Ensures that different shuffle order is used each epoch.
    for _, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        print(f'[{rank} - {torch.cuda.current_device()}] Memory allocated: {torch.cuda.memory_allocated()}')
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    # Reduce tensor 'ddp_loss' using sum op.
    # All processes will have identical copies following the call.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        # Print mean loss from rank 0 process only.
        mean_loss = ddp_loss[0] / ddp_loss[1]
        print(f'[Train] Epoch: {epoch} \tLoss: {mean_loss:.6f}')

def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)  # Track total loss, accuracy and number of samples.

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    # Reduce tensor 'ddp_loss' using sum op.
    # All processes will have identical copies following the call.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        mean_loss = ddp_loss[0] / ddp_loss[2]
        print(f'[Test] Loss: {mean_loss:.6f} \tAccuracy: {int(ddp_loss[1])}/{int(ddp_loss[2])} ({100 * ddp_loss[1] / ddp_loss[2]})')

def process_main(rank, world_size, args):
    setup_process(rank, world_size)

    # Create train/test dataloaders.
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)
    train_kwargs = {
        'batch_size': args.batch_size,
        'sampler': train_sampler
    }
    test_kwargs = {
        'batch_size': args.test_batch_size,
        'sampler': test_sampler
    }
    cuda_kwargs = {
        'num_workers': 2,
        'pin_memory': True,
        'shuffle': False
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Create timers.
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Create model wrapped in 'FSDP' to shard model parameters.
    torch.cuda.set_device(rank)
    model = Net().to(rank)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    # my_auto_wrap_policy = None
    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)

    # Create optimiser with LR scheduler.
    optimizer = Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train/test with timing.
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, train_sampler)
        print(f'[{rank} - {torch.cuda.current_device()}] Memory allocated: {torch.cuda.memory_allocated()}')
        test(model, rank, world_size, test_loader)
        scheduler.step()
    init_end_event.record()

    if rank == 0:
        time_s = init_start_event.elapsed_time(init_end_event) / 1000
        print(f'CUDA event elapsed time: {time_s}s.')
        print(f'{model}')

    teardown_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16384, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=16384, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    # WORLD_SIZE = 1

    mp.spawn(process_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
