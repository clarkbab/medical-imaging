import argparse
from functools import partial
import os
from re import I
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.multiprocessing as mp
from torch import nn
from torch.nn import NLLLoss
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

class Module(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.__loss = NLLLoss()
        self.__network = Net()
        self.__first_training_step = True

    def configure_optimizers(self):
        return Adadelta(self.trainer.model.parameters(), lr=1)

    # def configure_sharded_model(self):
    #     auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=10)
    #     # 'auto_wrap_policy' causes recursive layer wrapping using custom policy, 'device_id' ensures sharding happens
    #     # on GPU.
    #     self.__network = FSDP(self.__network, auto_wrap_policy=auto_wrap_policy, device_id=torch.distributed.get_rank())

    def forward(self, x):
        return self.__network(x)

    def training_step(self, batch, _):
        x, y = batch
        if self.__first_training_step:
            print(self.__network)
            self.__first_training_step = False
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y)
        print(f'[{torch.distributed.get_rank()} - {torch.cuda.current_device()}] Memory allocated: {torch.cuda.memory_allocated()}')
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y)
        return loss

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

    # Create train/test dataloaders.
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    train_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
    }
    test_kwargs = {
        'batch_size': args.test_batch_size,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Create model.
    model = Module()

    # Create trainer.
    trainer = Trainer(
        accelerator='gpu',
        devices=2,
        max_epochs=5,
        num_nodes=1,
        num_sanity_val_steps=0,
        precision=16,
        strategy='fsdp_native')

    # Fit model.
    trainer.fit(model, train_loader, val_loader)
