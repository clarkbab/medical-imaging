import os
from os.path import dirname as up
import pathlib
import shutil
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config
from mymi.regions import OldRegionNames

for_real = True
types = ['localiser', 'segmenter']
regions = OldRegionNames

for type in types:
    for region in regions:
        # Print model.
        id = f'{type}:{region}'
        print(f'model - {id}')

        # Get model folder.
        run_folder = os.path.join(config.directories.models, f'{type}-{region}')
        if not os.path.exists(run_folder):
            continue

        # Remove model.
        if for_real:
            print(f'\t{run_folder}')
            shutil.rmtree(run_folder)
        else:
            print(f'\t{run_folder}')
