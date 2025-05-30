import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.reporting.datasets.nifti import create_region_figures

fire.Fire(create_region_figures)

# Sample args:
# --dataset HN1 --region 'Parotid_L'
