import os
from typing import *

from mymi.dataset import NiftiDataset
from mymi import logging
from mymi.types import *
from mymi.utils import append_row

def get_custom_holdout_split(
    dataset: str,
    exists_only: bool = False) -> Union[Tuple[List[PatientID], List[PatientID], List[PatientID]], bool]:
    # Loads train/validate/test for 'split.csv' file.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'holdout-split.csv')
    if not os.path.exists(filepath):
        if exists_only:
            return False
        else:
            raise ValueError(f"No split file found for dataset '{dataset}'.")     
    elif exists_only:
        return True

    # Load split file.
    logging.info(f"Loading custom split 'holdout-split.csv'.")
    df = pd.read_csv(filepath)
    
    # Get splits.
    train_ids = df[df['split'] == 'train']['patient-id'].tolist()
    val_ids = df[df['split'] == 'validate']['patient-id'].tolist()
    test_ids = df[df['split'] == 'test']['patient-id'].tolist()

    return train_ids, val_ids, test_ids

def get_holdout_split(
    # Splits into train/validate/test.
    dataset: str,
    p_test: float = 0.2,
    p_val: float = 0.2,
    random_seed = 42,
    shuffle: bool = True,
    use_custom: bool = True) -> Tuple[List[PatientID], List[PatientID], List[PatientID]]:
    # Check for custom split.
    if use_custom and get_custom_holdout_split(dataset, exists_only=True):
        return get_custom_holdout_split(dataset)

    # Load patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(pat_ids)
        
    # Get splits numbers.
    p_train_total = 1 - p_test
    n_pats = len(pat_ids)
    n_train_total = int(p_train_total * n_pats)
    p_train = 1 - p_val
    n_train = int(p_train * n_train_total)
    n_val = n_train_total - n_train

    # Get patients.
    train_ids = pat_ids[:n_train]
    val_ids = pat_ids[n_train:n_train + n_val]
    test_ids = pat_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids

def get_holdout_split_regions(
    dataset,
    **kwargs) -> pd.DataFrame:
    # Get split.
    set = NiftiDataset(dataset)
    split_pat_ids = get_holdout_split(dataset, **kwargs)
    
    cols = {
        'split': str,
        'region': str,
        'count': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    # Count regions per split.
    set = NiftiDataset(dataset)
    splits = ['train', 'validate', 'test']
    for s, pat_ids in zip(splits, split_pat_ids):
        region_counts = {}
        for p in pat_ids:
            pat = set.patient(p)
            regions = pat.list_regions()
            for r in regions:
                if r in region_counts:
                    region_counts[r] += 1
                else:
                    region_counts[r] = 1
                    
        for r, c in sorted(region_counts.items()):
            data = {
                'split': s,
                'region': r,
                'count': c
            }
            df = append_row(df, data)
            
    return df
