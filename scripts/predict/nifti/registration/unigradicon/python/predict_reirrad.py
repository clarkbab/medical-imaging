import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.predictions.datasets.nifti import create_unigradicon_predictions

dataset = 'PMCC-REIRRAD'
model = 'unigradicon'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='RL:PMCC-REIRRAD',
    use_io=False,
)
create_unigradicon_predictions(dataset, model, **kwargs)
