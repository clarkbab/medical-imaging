import os
import pandas as pd
from tqdm import tqdm

from mymi import config
from mymi.loaders import Loader
from mymi.models import list_checkpoints
from mymi.regions import RegionNames
from mymi.utils import append_row, encode

def create_model_manifest() -> None:
    datasets = ('PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC')
    model_types = ['localiser', 'segmenter']
    model_subtypes = ['clinical', 'public', 'transfer']
    num_folds = 5
    num_trains = (5, 10, 20, 50, 100, 200, None)
    regions = RegionNames
    test_folds = tuple(range(num_folds))

    cols = {
        'name': str,
        'run': str,
        'checkpoint': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add public models.
    for model_type in tqdm(model_types):
        for region in tqdm(regions, leave=False):
            name = f'{model_type}-{region}'
            for model_subtype in model_subtypes:
                if model_subtype == 'public':
                    run = 'public-1gpu-150epochs'
                    ckpts = list_checkpoints(name, run)
                    for ckpt in ckpts:
                        data = {
                            'name': name,
                            'run': run,
                            'checkpoint': ckpt
                        }
                        df = append_row(df, data)
                elif model_type == 'segmenter':
                    for test_fold in test_folds:
                        for num_train in num_trains:
                            # Check model exists.
                            tl, vl, _ = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=test_fold)
                            num_train_max = len(tl) + len(vl)
                            if num_train != None and num_train > num_train_max:
                                continue

                            run = f'{model_subtype}-fold-{test_fold}-samples-{num_train}'
                            ckpts = list_checkpoints(name, run)
                            for ckpt in ckpts:
                                data = {
                                    'name': name,
                                    'run': run,
                                    'checkpoint': ckpt
                                }
                                df = append_row(df, data)

    # Save manifest.
    df = df.astype(cols)
    config.save_csv(df, 'model-manifest.csv', overwrite=True) 
    
def load_model_manifest():
    return config.load_csv('model-manifest.csv')
