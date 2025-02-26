import pandas as pd

from mymi.typing import *
from mymi.utils import *

def load_training_metrics(
    model: ModelName) -> pd.DataFrame:
    # Load data.
    filepath = os.path.join(config.directories.models, model[0], model[1], 'training-metrics.csv')
    return load_csv(filepath)
