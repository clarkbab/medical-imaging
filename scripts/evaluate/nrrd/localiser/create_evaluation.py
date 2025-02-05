import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluations.datasets.nrrd import create_localiser_evaluation

fire.Fire(create_localiser_evaluation)
