from typing import List, Optional, Union

from .adaptive_loader import AdaptiveLoader
from .loader import Loader
from .multi_loader import MultiLoader
from .multi_loader_convergence import MultiLoaderConvergence
from .multi_loader_v2 import MultiLoaderV2
from .reg_loader import RegLoader

def get_n_train_max(
    datasets: Union[str, List[str]],
    region: str,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> int:
    tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
    n_train = len(tl) + len(vl)
    return n_train

def get_n_test(
    datasets: Union[str, List[str]],
    region: str,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> int:
    _, _, tsl = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
    n_test = len(tsl)
    return n_test
