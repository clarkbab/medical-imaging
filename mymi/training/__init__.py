import os
import pandas as pd
from re import match
import torch
from typing import List, Optional, Union
from tqdm import tqdm

from mymi import config
from mymi.models import replace_ckpt_alias
from mymi.types import ModelName
from mymi.utils import append_row, arg_to_list

# from .auto_encoder_2d import train_auto_encoder_2d
from .localiser import train_localiser
# from .segmenter_2d import train_segmenter_2d
from .multi_segmenter import train_multi_segmenter
from .multi_segmenter_pytorch import train_multi_segmenter_pytorch
from .localiser_replan import train_localiser_replan
from .segmenter import train_segmenter
from .segmenter_test import train_segmenter_test
from .segmenter_parallel import train_segmenter_parallel
from .memory_test import train_memory_test

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
            n_epochs = state['epoch']
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
