import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluations.datasets.nifti import create_multi_segmenter_heatmap_evaluation

fire.Fire(create_multi_segmenter_heatmap_evaluation)
