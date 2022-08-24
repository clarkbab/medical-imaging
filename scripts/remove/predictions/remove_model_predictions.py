import os
from os.path import dirname as up
import pandas as pd
import pathlib
from tqdm import tqdm
import shutil
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config
from mymi.loaders import Loader
from mymi.models.systems import Localiser, Segmenter
from mymi.regions import RegionNames

for_real = True
datasets_for_loader = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
datasets = ['PMCC-HN-TEST', 'PMCC-HN-TRAIN']
regions = RegionNames
models = ['public', 'clinical', 'transfer']
num_folds = 5
num_trains = [5, 10, 20, 50, 100, 200, None]
test_folds = [0, 1, 2, 3, 4]

# Load localiser checkpoints.
loc_ckpts = {}
for region in regions:
    ckpt = Localiser.replace_checkpoint_aliases(f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')[2]
    loc_ckpts[region] = ckpt

for region in regions:
    for model in models:
        if model == 'public':
            # Get model run.
            seg_run = 'public-1gpu-150epochs'
            print(f'region:{region}, model:{seg_run}')

            for dataset in datasets:
                # Get path.
                run_path = os.path.join(config.directories.datasets, 'nifti', dataset, 'predictions', 'segmenter', f'localiser-{region}', 'public-1gpu-150epochs', loc_ckpts[region], f'segmenter-{region}', seg_run)
                if os.path.exists(run_path):
                    if for_real:
                        print(f'\t{run_path}')
                        shutil.rmtree(run_path)
                    else:
                        print(f'\t{run_path}')
        else:
            for test_fold in test_folds:
                # Get test cases per dataset.
                tl, vl, _ = Loader.build_loaders(datasets_for_loader, region, num_folds=num_folds, test_fold=test_fold)
                num_train_max = len(tl) + len(vl)
            
                for num_train in num_trains:
                    # Check that number of training cases are available.
                    if num_train is not None and num_train > num_train_max:
                        continue

                    # Get model run.
                    seg_run = f'{model}-fold-{test_fold}-samples-{num_train}'
                    print(f'region:{region}, model:{seg_run}')

                    for dataset in datasets:
                        # Get path.
                        run_path = os.path.join(config.directories.datasets, 'nifti', dataset, 'predictions', 'segmenter', f'localiser-{region}', 'public-1gpu-150epochs', loc_ckpts[region], f'segmenter-{region}', seg_run)
                        if os.path.exists(run_path):
                            if for_real:
                                print(f'\t{run_path}')
                                shutil.rmtree(run_path)
                            else:
                                print(f'\t{run_path}')
