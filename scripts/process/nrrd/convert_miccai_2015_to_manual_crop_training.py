import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.nrrd import convert_miccai_2015_to_manual_crop_training

fire.Fire(convert_miccai_2015_to_manual_crop_training)
