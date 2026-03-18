from colorlog import ColoredFormatter
import inspect
import logging as python_logging
from typing import *

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LEVEL_MAP = {
    10: 'DEBUG',
    20: 'INFO',
    30: 'WARNING',
    40: 'ERROR',
    50: 'CRITICAL'
}
LOG_FORMAT = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

logger = None

def config(level: str) -> None:
    global logger

    # Create logger and set level.
    logger = python_logging.getLogger('MYMI')
    level = getattr(python_logging, level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Logging level '{level}' not valid.")
    logger.setLevel(level)

    # Create console handler and set level.
    ch = python_logging.StreamHandler()
    ch.setLevel(level)

    # Add formatter to console handler.
    formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
    ch.setFormatter(formatter)

    # Remove old handlers.
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Add console handler to logger.
    logger.addHandler(ch)

def level():
    return LEVEL_MAP[logger.level]

def debug(*args, **kwargs):
    global logger
    return logger.debug(*args, **kwargs)

def info(*args, **kwargs):
    global logger
    return logger.info(*args, **kwargs)

def warning(*args, **kwargs):
    global logger
    return logger.warning(*args, **kwargs)

def error(*args, **kwargs):
    global logger
    return logger.error(*args, **kwargs)

def critical(*args, **kwargs):
    global logger
    return logger.critical(*args, **kwargs)

def arg_log(
    action: str,
    arg_names: Union[str, List[str]],
    arg_vals: Union[Any, List[Any]]) -> None:
    message = action + ' with ' + ', '.join([f"{arg_name}={arg_val}" for arg_name, arg_val in zip(arg_names, arg_vals)]) + '.'
    info(message)

def _format_arg(val: Any) -> str:
    """Format a value for logging, abbreviating arrays and tensors to type+shape."""
    import numpy as np
    try:
        import torch
        is_tensor = isinstance(val, torch.Tensor)
    except ImportError:
        is_tensor = False
    if isinstance(val, np.ndarray):
        return f"ndarray(shape={val.shape}, dtype={val.dtype})"
    if is_tensor:
        return f"Tensor(shape={tuple(val.shape)}, dtype={val.dtype})"
    return repr(val)

def log_args(message: str = '') -> None:
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    arg_info = inspect.getargvalues(frame)
    parts = []
    for name in arg_info.args:
        parts.append(f"{name}={_format_arg(arg_info.locals[name])}")
    if arg_info.varargs and arg_info.locals.get(arg_info.varargs):
        for val in arg_info.locals[arg_info.varargs]:
            parts.append(_format_arg(val))
    if arg_info.keywords and arg_info.locals.get(arg_info.keywords):
        for k, v in arg_info.locals[arg_info.keywords].items():
            parts.append(f"{k}={_format_arg(v)}")
    fn_str = f"{func_name}({', '.join(parts)})"
    if message:
        info(f"{message}: {fn_str}")
    else:
        info(fn_str)

# Default config.
config('info')
