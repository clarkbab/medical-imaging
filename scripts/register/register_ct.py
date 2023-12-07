import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from mymi.transforms.dataset.nifti import create_registered_ct

fire.Fire(create_registered_ct)
