import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.nifti.nifti import combine_segmenter_predictions_from_all_patients

fire.Fire(combine_segmenter_predictions_from_all_patients)
