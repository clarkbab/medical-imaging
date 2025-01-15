import os
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import *

from mymi.dataset import NiftiDataset
from mymi.dataset.training import recreate
from mymi.loaders import get_holdout_split
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import resample
from mymi.types import *
from mymi.utils import append_row

def convert_to_training_holdout(
    dataset: str,
    dest_dataset: Optional[str] = None,
    regions: Optional[PatientRegions] = 'all',
    spacing: Optional[ImageSpacing3D] = None,
    **kwargs) -> None:
    start = time()
    logging.arg_log('Converting NIFTI dataset to holdout TRAINING', ('dataset', 'spacing', 'regions'), (dataset, spacing, regions))

    # Get regions.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })

    # Create training dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    dest_set = recreate(dest_dataset)

    # Write params.
    cols = {
        'param': str,
        'value': str
    }
    df = pd.DataFrame(columns=cols.keys())

    params = [
        { 'param': 'type', 'value': 'holdout' },
        { 'param': 'spacing', 'value': str(spacing) },
        { 'param': 'regions', 'value': str(regions) }
    ]
    for p in params:
        df = append_row(df, p)
    filepath = os.path.join(dest_set.path, 'params.csv')
    df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(regions=regions)

    cols = {
        'split': str,
        'sample-id': int,
        'origin-dataset': str,
        'origin-patient-id': str,
        'origin-study-id': str,
    }
    df = pd.DataFrame(columns=cols.keys())

    # Write each split to the dataset.
    pat_ids = get_holdout_split(dataset, **kwargs)
    splits = ['train', 'validate', 'test']
    sample_id_counter = 0
    for s, ps in tqdm(zip(splits, pat_ids)):
        for p in tqdm(ps, leave=False):
            sample_id = f"{sample_id_counter:03}"
            
            # Load sample data.
            pat = set.patient(p)
            study = pat.default_study
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            region_data = study.region_data(regions=regions)

            # Resample data.
            if spacing:
                ct_data = resample(ct_data, spacing=ct_spacing, output_spacing=spacing)
                for r, d in region_data.items():
                    region_data[r] = resample(d, spacing=ct_spacing, output_spacing=spacing)

            # Save input data.
            filepath = os.path.join(dest_set.path, 'data', s, 'inputs', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=ct_data)
            
            # Create/save output data.
            n_channels = len(regions) + 1
            label = np.zeros((n_channels, *ct_data.shape), dtype=bool)
            mask = np.zeros((n_channels), dtype=bool)
            for r, d in region_data.items():
                channel = regions.index(r) + 1
                label[channel] = d
                mask[channel] = True

            # Add background if all foreground classes present. 
            if pat.has_regions(regions):
                label[0] = np.invert(label.any(axis=0))
                mask[0] = True

            # Save label data.
            filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=label)
            filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}_mask.npz")
            np.savez_compressed(filepath, data=mask, exist_ok=True)

            # Add index entry.
            data = {
                'split': s,
                'sample-id': sample_id,
                'origin-dataset': dataset,
                'origin-patient-id': p,
                'origin-study-id': study.id,
            }
            df = append_row(df, data)

            sample_id_counter += 1

    # Write index.
    df = df.astype(cols)
    filepath = os.path.join(dest_set.path, 'index.csv')
    df.to_csv(filepath, index=False)

    # Record processing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    path = os.path.join(dest_set.path, f'__CONVERT_FROM_NIFTI_END_MINS_{mins}__')
    Path(path).touch()
