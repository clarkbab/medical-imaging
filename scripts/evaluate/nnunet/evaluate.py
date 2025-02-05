import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluations.datasets.nifti.custom import create_nnunet_single_region_evaluation

fire.Fire(create_nnunet_single_region_evaluation)
