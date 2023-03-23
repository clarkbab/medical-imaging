import os
from os.path import dirname as up
import pathlib
import shutil
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi import config
from mymi.regions import OldRegionNames

for_real = False
types = ['segmenter']
regions = OldRegionNames
runs = []
folds = [0, 1, 2, 3, 4]
n_trains = [5, 10, 20, 50, 100, 200, None]
for fold in folds:
    for n_train in n_trains:
        runs.append(f'transfer-fold-{fold}-samples-{n_train}')

for type in types:
    for region in regions:
        # Print model.
        id = f'{type}:{region}'
        print(f'model - {id}')

        for run in runs:
            # Get run folder.
            run_folder = os.path.join(config.directories.models, f'{type}-{region}', run)
            if not os.path.exists(run_folder):
                continue

            # Remove run.
            if for_real:
                print(f'\t{run_folder}')
                shutil.rmtree(run_folder)
            else:
                print(f'\t{run_folder}')
