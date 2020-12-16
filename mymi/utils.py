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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logging.info('Running on GPU!')
    else:
        logging.info('Using CPU.')
    return device

def configure_logging(log_level):
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}.")
    log_format = "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(log_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(handlers=[stream], level=log_level)

