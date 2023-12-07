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

for_real = False
datasets_for_loader = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
datasets = ['PMCC-HN-TEST', 'PMCC-HN-TRAIN']
regions = RegionList.PMCC
test_folds = list(range(5))
n_folds = 5
n_trains = [5, 10, 20, 50, 100, 200, None]
n_trains = [5, 10]
use_seg_run = True

for region in regions:
    for test_fold in test_folds:
        n_train_max = get_n_train_max(datasets_for_loader, region, n_folds=n_folds, test_fold=test_fold)
    
        for n_train in n_trains:
            # Check that number of training cases are available.
            if n_train is not None and n_train > n_train_max:
                continue

            localiser = (f'localiser-{region}', f'clinical-fold-{test_fold}-samples-{n_train}', 'best')
            localiser = replace_ckpt_alias(localiser)

            # Get model run.
            segmenter = (f'segmenter-{region}-v2', f'{loc_model}-fold-{test_fold}-samples-{n_train}', 'best')
            segmenter = replace_ckpt_alias(segmenter)

            # Get test cases per dataset.
            _, _, tsl = Loader.build_loaders(datasets_for_loader, region, n_folds=n_folds, test_fold=test_fold, use_seg_run=use_seg_run)

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
