import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.dataset.nifti.nifti import create_excluded_brainstem
fire.Fire(create_excluded_brainstem)
