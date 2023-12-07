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
from mymi.loaders import get_n_train_max
from mymi.models import replace_ckpt_alias
from mymi.regions import RegionList

for_real = True
datasets_for_loader = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
datasets = ['PMCC-HN-TEST', 'PMCC-HN-TRAIN']
regions = RegionList.PMCC
regions = ['Parotid_L', 'Parotid_R']
test_folds = list(range(5))
models = ['clinical-v2']
n_folds = 5
n_trains = [5, 10, 20, 50, 100, 200, None]
n_trains = [5, 10]

for region in regions:
    for model in models:
        if model == 'public':
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')
            localiser = replace_ckpt_alias(localiser)

            # Get model run.
            seg_run = 'public-1gpu-150epochs'
            print(f'region:{region}, model:{seg_run}')

            for dataset in datasets:
                # Get path.
                run_path = os.path.join(config.directories.datasets, 'nifti', dataset, 'predictions', 'segmenter', *localiser, f'segmenter-{region}', seg_run)
                if os.path.exists(run_path):
                    if for_real:
                        print(f'\t{run_path}')
                        shutil.rmtree(run_path)
                    else:
                        print(f'\t{run_path}')
        else:
            for test_fold in test_folds:
                n_train_max = get_n_train_max(datasets_for_loader, region, n_folds=n_folds, test_fold=test_fold)
            
                for n_train in n_trains:
                    # Check that number of training cases are available.
                    if n_train is not None and n_train > n_train_max:
                        continue

                    # Get localiser.
                    if model == 'clinical-v2':
                        loc_model = 'clinical'
                    else:
                        loc_model = model
                    localiser = (f'localiser-{region}', f'{loc_model}-fold-{test_fold}-samples-{n_train}', 'best')
                    localiser = replace_ckpt_alias(localiser)

                    # Get model run.
                    segmenter = (f'segmenter-{region}-v2', f'{loc_model}-fold-{test_fold}-samples-{n_train}', 'best')
                    segmenter = replace_ckpt_alias(segmenter)

                    # Get test cases per dataset.
                    _, _, tsl = Loader.build_loaders(datasets_for_loader, region, n_folds=n_folds, test_fold=test_fold)

                    for desc_b in tsl:
                        for desc in desc_b:
                            dataset, pat_id = desc.split(':')

                            # Get path.
                            filepath = os.path.join(config.directories.predictions, 'data', 'segmenter', dataset, pat_id, *localiser, *segmenter, 'pred.npz')
                            if not os.path.exists(filepath):
                                # raise ValueError(f"Filepath '{filepath}' not found.")
                                pass

                            if os.path.exists(filepath):
                                if for_real:
                                    print(f'\t{filepath}')
                                    os.remove(filepath)
                                else:
                                    print(f'\t{filepath}')
