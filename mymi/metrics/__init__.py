from .batch import *
from .dice import *
from .distances import *
from .intensity import *
from .registration import *
from .volume import *

# In which direction does the metric improve?
# Higher is better (True) or lower is better (False).
def higher_is_better(metric: str) -> bool:
    if 'apl-mm-tol-' in metric:
        return False
    if metric == 'dice':
        return True
    if 'dm-surface-dice-tol-' in metric:
        return True
    if 'hd' in metric: 
        return False
    if metric == 'msd':
        return False
