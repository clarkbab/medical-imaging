import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.gradcam.dataset.nifti import create_heatmap

fire.Fire(create_heatmap)
