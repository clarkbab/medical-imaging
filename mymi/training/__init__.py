import os
import pandas as pd
from pandas import DataFrame
from re import match
import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from mymi import config
from mymi.models import replace_ckpt_alias
from mymi.typing import ModelName
from mymi.utils import append_row, arg_to_list

from .adaptive_segmenter import train_adaptive_segmenter
from .localiser import train_localiser
from .multi_segmenter import train_multi_segmenter
from .multi_segmenter_convergence import train_multi_segmenter_convergence
from .multi_segmenter_pytorch import train_multi_segmenter_pytorch
from .localiser_replan import train_localiser_replan
from .segmenter import *
from .segmenter_test import train_segmenter_test
from .segmenter_parallel import train_segmenter_parallel
from .memory_test import train_memory_test

def load_run_manifest(model: Tuple[str, str]) -> DataFrame:
    filepath = os.path.join(config.directories.runs, *model)
    latest_run = list(sorted(os.listdir(filepath)))[-1]
    filepath = os.path.join(filepath, latest_run, 'multi-loader-manifest.csv')
    df = pd.read_csv(filepath)
    return df

def get_n_epochs(
    model: Union[ModelName, List[ModelName]],
    **kwargs) -> pd.DataFrame:
    models = arg_to_list(model, tuple)
    
    cols = {
        'model': str,
        'run': str,
        'ckpt': str,
        'exists': bool,
        'n-epochs': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    models = tqdm(models) if len(models) > 1 else models
    for model in models:
        try:
            model = replace_ckpt_alias(model, **kwargs)
            filepath = os.path.join(config.directories.models, *model[:2], f'{model[2]}.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_epochs = state['epoch'] + 1       # Starts at 0.
            exists = True
        except ValueError as e:
            exists = False
            n_epochs = 0
        
        data = {
            'model': model[0],
            'run': model[1],
            'ckpt': model[2],
            'exists': exists,
            'n-epochs': n_epochs
        }
        df = append_row(df, data)
            
    return df
