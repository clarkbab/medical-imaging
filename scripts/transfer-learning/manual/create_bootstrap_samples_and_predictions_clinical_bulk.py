import fire
import os
import subprocess
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.regions import RegionList
from mymi.transfer_learning import create_bootstrap_samples_and_predictions

dataset = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
all_regions = RegionList.PMCC
regions = [13]
model_types = ['clinical-v2', 'transfer']
metrics = ['apl-mm-tol-{tol}', 'dice', 'dm-surface-dice-tol-{tol}', 'hd', 'hd-95', 'msd']
stats = ['mean', 'q1', 'q3']

for region_id in regions:
    region = all_regions[region_id]
    for model_type in model_types:
        for metric in metrics:
            for stat in stats:
                create_bootstrap_samples_and_predictions(dataset, region, model_type, metric, stat)
