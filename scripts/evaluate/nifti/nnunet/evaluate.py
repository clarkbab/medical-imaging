import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluation.dataset.nifti import create_nnunet_multi_segmenter_evaluation

fire.Fire(create_nnunet_multi_segmenter_evaluation)
