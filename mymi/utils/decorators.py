from functools import wraps
from typing import *

from mymi.typing import *

from .args import arg_to_list

def alias_kwargs(aliases: Union[Tuple[str, str], List[Tuple[str, str]]]) -> Callable:
    aliases = arg_to_list(aliases, tuple)
    alias_map = dict(aliases)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for shortcut, full_name in alias_map.items():
                if shortcut in kwargs:
                    kwargs[full_name] = kwargs.pop(shortcut)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def handle_non_spatial_dims(
    spatial_fn: Callable,
    data: Union[ImageArrays, ImageTensors],
    *args, **kwargs) -> ImageArray:
    datas = arg_to_list(data, (ImageArray, ImageTensor))
    outputs = []
    for data in datas:
        n_dims = len(data.shape)
        if n_dims in (2, 3):
            output = spatial_fn(data, *args, **kwargs)
        elif n_dims == 4:
            size = data.shape
            if size[0] > size[-1]:
                logging.warning(f"Channels dimension should come first when resampling 4D. Got shape {size}, is this right?")
            os = []
            for d in data:
                d = spatial_fn(d, *args, **kwargs)
                os.append(d)
            output = __stack(os, axis=0)
        elif n_dims == 5:
            size = data.shape
            if size[0] > size[-1]:
                logging.warning(f"Batch dimension should come first when resampling 5D. Got shape {size}, is this right?")
            if size[1] > size[-1]:
                logging.warning(f"Channel dimension should come second when resampling 5D. Got shape {size}, is this right?")
            bs = []
            for batch_item in data:
                ocs = []
                for channel_data in batch_item:
                    oc = spatial_fn(channel_data, *args, **kwargs)
                    ocs.append(oc)
                ocs = __stack(ocs, axis=0)
                bs.append(ocs)
            output = __stack(bs, axis=0)
        else:
            raise ValueError(f"Data should have (2, 3, 4, 5) dims, got {n_dims}.")

        outputs.append(output)

    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs

def __stack(
    data: Union[List[ImageArray], List[ImageTensor]],
    axis: int = 0) -> Union[ImageArray, ImageTensor]:
    stack_fn = lambda x, a: np.stack(x, axis=a) if isinstance(x[0], np.ndarray) else torch.stack(x, dim=a)
    data = stack_fn(data, axis)
    return data
