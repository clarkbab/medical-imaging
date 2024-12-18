import os
import subprocess
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.prediction.dataset.nifti.custom import predict_unigradicon

dataset = 'PMCC-REIRRAD'
model = 'UNIGRADICON-IO'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='RL:PMCC-REIRRAD',
    use_io=True,
)
predict_unigradicon(dataset, model, **kwargs)
