import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.nnunet import convert_predictions_to_nifti_single_region
fire.Fire(convert_predictions_to_nifti_single_region)
