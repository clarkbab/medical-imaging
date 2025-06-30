import pandas as pd

from mymi.typing import *
from mymi.utils import *

def replace_metrics(
    model: ModelName,
    replace: Dict[str, str]) -> None:
    filepath = os.path.join(config.directories.models, model[0], model[1], 'training-metrics.csv')
    df = load_files_csv(filepath)
    df['metric'] = df['metric'].rename(replace)
    save_csv(df, filepath)
