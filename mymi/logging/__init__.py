from colorlog import ColoredFormatter
import logging

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
    logger = logging.getLogger('MYMI')
    level = getattr(logging, level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Logging level '{level}' not valid.")
    logger.setLevel(level)

    # Create console handler and set level.
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Add formatter to console handler.
    formatter = ColoredFormatter(LOG_FORMAT, DATE_FORMAT)
    ch.setFormatter(formatter)

    # Remove old handlers.
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # Add console handler to logger.
    logger.addHandler(ch)

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

# Default config.
config('info')
