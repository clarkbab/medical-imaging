from mymi.transforms.crop import crop_foreground_vox
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Literal, Optional, Union

from mymi import config
from mymi.datasets import NrrdDataset
from mymi.geometry import get_box, centre_of_extent, extent_mm
from mymi.metrics import dice, distances, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi.models.lightning_modules import Segmenter
from mymi import logging
from mymi.predictions.nrrd import load_multi_segmenter_prediction_dict, load_segmenter_predictions
from mymi.regions import get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.typing import ModelName, Regions
from mymi.utils import append_row, arg_to_list, encode
