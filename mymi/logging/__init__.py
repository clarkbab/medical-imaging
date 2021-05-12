from colorlog import ColoredFormatter
import logging
from typing import *

def config(level: str) -> None:
    level = getattr(logging, level.upper(), None)
    assert isinstance(level, int)
    log_format = "%(log_color)s%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ColoredFormatter(log_format, date_format)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logging.basicConfig(handlers=[stream], level=level)

def info(*args, **kwargs):
    return logging.info(*args, **kwargs)
