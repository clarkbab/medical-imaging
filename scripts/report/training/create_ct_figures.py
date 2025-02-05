import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.reporting.datasets.training import create_ct_figures_report

fire.Fire(create_ct_figures_report)

# Sample args:
# --dataset HN1 --clear_cache True --regions "('Parotid_L','Parotid_R')"
