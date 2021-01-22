from colorlog import ColoredFormatter
import gc
import logging
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_ct(*args):
    """
    input: required.
    mask: optional.
    """
    # Get data.
    data = args[0]
    mask = args[1] if len(args) >= 2 else None
    pred = args[2] if len(args) >= 3 else None

    # Define first plotting axis.
    if pred is None:
        plt.figure(figsize=(8, 8))
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 16), sharey=True)

    # Remove "channel" dimension if present.
    if data.shape[0] == 1:
        data = data.squeeze(0)

    # Plot CT.
    ax = ax1 if pred is not None else plt
    ax.imshow(np.transpose(data), cmap='gray')
    
    # Plot mask.
    colours = [(1.0, 1.0, 1.0, 0), (0.12, 0.47, 0.70, 1.0)]
    mask_cmap = ListedColormap(colours)
    if mask is not None:
        ax.imshow(np.transpose(mask), cmap=mask_cmap)

    # Plot prediction.
    if pred is not None:
        ax2.imshow(np.transpose(data), cmap='gray')
        ax2.imshow(np.transpose(pred), cmap=mask_cmap)

def plot_batch(*args, **kwargs):
    """
    effect: plots a batch of images.
    """
    # Get options.
    figsize_mul = kwargs['figsize'] if 'figsize' in kwargs else 8

    # Get image data.
    data = image_data(*args, **kwargs)
    num_images = len(data)

    # Get keyword arguments.
    axis = 'on' if 'axis' in kwargs and kwargs['axis'] else 'off'

    figsize = (figsize_mul, num_images * figsize_mul)
    _, axs = plt.subplots(num_images, figsize=figsize)
    if num_images == 1: axs = [axs]

    for i in range(num_images):
        axs[i].axis(axis)
        axs[i].imshow(np.transpose(data[i]))
        
    plt.show()

def image_data(*args, **kwargs):
    """
    returns: an array of image data.
    """
    # Parse arguments.
    input = args[0].cpu().float()
    mask = args[1].cpu() if len(args) > 1 else None
    pred = args[2].detach().cpu().argmax(dim=1) if len(args) > 2 else None
    num_images = kwargs['num_images'] if 'num_images' in kwargs else input.shape[0]

    # Check if input has 'channel' dimension.
    if len(input.shape) == 4:
        input = input.squeeze(1)

    # Scale CT data.
    image_data = input[:num_images]
    min, max = torch.amin(image_data, dim=(1, 2)), torch.amax(image_data, dim=(1, 2))
    denom = max - min
    denom[denom == 0] = 1e-10
    image_data = (image_data - min.view(len(min), 1, 1)) / denom.view(len(denom), 1, 1)
    image_data = torch.cat(3 * [image_data.unsqueeze(1)], dim=1)

    # Add prediction.
    if pred is not None:
        pred_data = pred[:num_images]
        color = (0.12, 0.47, 0.7)
        pred_data = torch.stack([c * pred_data for c in color], dim=1)
        image_data[pred_data != 0] = pred_data[pred_data != 0]

    # Add mask data.
    if mask is not None:
        mask_data = mask[:num_images]
        color = (1., 1., 1e-5)
        mask_data = binary_perimeter(mask_data).float()
        mask_data = torch.stack([c * mask_data for c in color], dim=1)
        image_data[mask_data != 0] = mask_data[mask_data != 0]

    return image_data

def pretty_size(size):
    assert isinstance(size, torch.Size)
    return ' x '.join(map(str, size))

def dump_tensors(gpu_only=True):
    """
    Prints list of Tensors that are tracked by the garbage collector.
    """
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(f"{type(obj).__name__}:{' GPU' if obj.is_cuda else ''}{' pinned' if obj.is_pinned else ''} {pretty_size(obj.size())}")
                    total_size += obj.numel()
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        print(f"{type(obj).__name__} -> {type(obj.data).__name__}:{' GPU' if obj.is_cuda else ''}{' pinned' if obj.data.is_pinned else ''}{' grad' if obj.requires_grad else ''}{' volatile' if obj.volatile else ''} {pretty_size(obj.data.size())}")
                        total_size += obj.data.numel()
        except Exception as e:
            pass
    print(f"Total size: {total_size}")

def configure_device():
    num_devices = torch.cuda.device_count()
    logging.info(f"Found {num_devices} device/s:")
    for i in range(num_devices):
        logging.info(f"\t{torch.cuda.get_device_name(i)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logging.info('Running on GPU!')
    else:
        logging.info('Using CPU.')
    return device

def configure_logging(log_level):
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {log_level}.")
    log_format = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(log_format, date_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(handlers=[stream], level=log_level, force=True)

def binary_perimeter(mask):
    mask_perimeter = torch.zeros_like(mask, dtype=bool)
    b_dim, x_dim, y_dim = mask.shape
    for b in range(b_dim):
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (mask[b, i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and mask[b, i - 1, j] == 0 or 
                    i != x_dim - 1 and mask[b, i + 1, j] == 0 or
                    j != 0 and mask[b, i, j - 1] == 0 or
                    j != y_dim - 1 and mask[b, i, j + 1] == 0)):
                    mask_perimeter[b, i, j] = 1
    return mask_perimeter
