import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.prediction.dataset.nrrd import create_all_localiser_predictions

fire.Fire(create_all_localiser_predictions)
