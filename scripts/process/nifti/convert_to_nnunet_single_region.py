import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.nifti import convert_to_nnunet_single_region

fire.Fire(convert_to_nnunet_single_region)
