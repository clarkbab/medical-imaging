import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.predictions.nifti.segmentation.segmentation import create_multi_segmenter_predictions_nnunet_bootstrap

fire.Fire(create_multi_segmenter_predictions_nnunet_bootstrap)
