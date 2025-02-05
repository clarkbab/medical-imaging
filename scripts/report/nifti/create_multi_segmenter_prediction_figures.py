import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.reporting.datasets.nifti import create_multi_segmenter_prediction_figures

fire.Fire(create_multi_segmenter_prediction_figures)
