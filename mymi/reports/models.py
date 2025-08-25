import os
import pandas as pd
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.regions import RegionNames
from mymi.typing import *
from mymi.utils import *

def load_training_metrics(
    model: ModelName) -> pd.DataFrame:
    filepath = os.path.join(config.directories.models, model[0], model[1], 'training-metrics.csv')
    return load_files_csv(filepath)

def list_checkpoints(
    name: str,
    run: str) -> List[str]:
    ckptspath = os.path.join(config.directories.models, name, run)
    ckpts = list(sorted([c.replace('.ckpt', '') for c in os.listdir(ckptspath)]))
    return ckpts
