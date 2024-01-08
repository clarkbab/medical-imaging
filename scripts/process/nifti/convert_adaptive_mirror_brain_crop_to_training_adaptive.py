import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.dataset.nifti import convert_adaptive_mirror_brain_crop_to_training_adaptive

fire.Fire(convert_adaptive_mirror_brain_crop_to_training_adaptive)
