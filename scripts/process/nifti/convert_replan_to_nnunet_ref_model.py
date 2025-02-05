import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.nifti.nifti import convert_replan_to_nnunet_ref_model

fire.Fire(convert_replan_to_nnunet_ref_model)
