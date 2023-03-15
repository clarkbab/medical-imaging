import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.reporting.dataset.nifti import create_ct_summary

fire.Fire(create_ct_summary)
