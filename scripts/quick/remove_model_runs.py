import os
from os.path import dirname as up
import pathlib
import shutil
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi import config
from mymi.regions import RegionNames

dry_run = False
keep = 2
types = ['segmenter']
regions = RegionNames
runs = ['1gpu-200epochs', '4gpu-200epochs', '4gpu-400epochs', 'test-centre-patchloader']

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
            if dry_run:
                print(f'\t{run_folder}')
            else:
                print(f'\t{run_folder}')
                shutil.rmtree(run_folder)
