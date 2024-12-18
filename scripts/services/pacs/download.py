import os
from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi.download.pacs import download_dicoms

download_dicoms()
