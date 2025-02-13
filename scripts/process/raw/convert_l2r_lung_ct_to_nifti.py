import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.raw import convert_l2r_lung_ct_to_nifti

fire.Fire(convert_l2r_lung_ct_to_nifti)
