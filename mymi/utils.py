from colorlog import ColoredFormatter
import gc
import logging
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch

from mymi import dataset

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

def filterOnPatID(pat_id):
    """
    returns: a function to filter based on 'pat_id' kwarg.
    args:
        pat_id: the passed 'pat_id' kwarg.
    """
    def fn(id):
        if (pat_id == 'all' or 
            (isinstance(pat_id, str) and id == pat_id) or
            ((isinstance(pat_id, list) or isinstance(pat_id, tuple)) and id in pat_id)):
            return True
        else:
            return False

    return fn

def filterOnRegion(region):
    """
    returns: a function to filter based on 'regions' kwarg.
    args:
        region: the passed 'region' kwarg.
    """
    def fn(id):
        # Load patient regions.
        pat_regions = dataset.patient_regions(id).region.to_numpy()

        if (region == 'all' or
            (isinstance(region, str) and region in pat_regions) or
            ((isinstance(region, list) or isinstance(region, tuple)) and len(np.intersect1d(region, pat_regions)) != 0)):
            return True
        else:
            return False

    return fn

def stringOrSorted(obj):
    """
    returns: no-op if obj is a string, else sorted tuple.
    args:
        obj: a string or iterable.
    """
    return obj if isinstance(obj, str) else tuple(sorted(obj))
