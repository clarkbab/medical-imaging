import numpy as np
import os
from tqdm import tqdm
from typing import *

from mymi.datasets.nifti import recreate as recreate_nifti
from mymi.datasets.nrrd import NrrdDataset
from mymi.geometry import extent_mm
from mymi import logging
from mymi.predictions.datasets.nrrd import load_localiser_prediction
from mymi.processing import write_flag
from mymi.transforms import crop_mm_3D
from mymi.typing import *
from mymi.utils import save_nifti

def convert_to_brain_crop(
    dataset: str,
    dest_dataset: str,
    margins_mm: BoxMM3D = [(50, 100, 120), (50, 0, 30)]) -> None:
    logging.arg_log(f'Converting NRRD dataset to brain crop', ('dataset', 'dest_dataset', margins_mm), (dataset, dest_dataset, margins_mm))

    # Create the dataset.
    set = NrrdDataset(dataset)
    dest_set = recreate_nifti(dest_dataset)

    pat_ids = set.list_patients()
    localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
    for p in tqdm(pat_ids):
        # Load brain prediction.
        pat = set.patient(p)
        spacing = pat.ct_spacing
        offset = pat.ct_offset
        brain_pred = load_localiser_prediction(dataset, p, localiser)
        brain_extent = extent_mm(brain_pred, spacing, offset)

        # Add margins.
        crop_min, crop_max = brain_extent
        crop_min = np.array(crop_min) - margins_mm[0]
        crop_max = np.array(crop_max) + margins_mm[1]
        crop_mm = (crop_min, crop_max)

        # Apply crop to CT data.
        ct_data = pat.ct_data
        ct_data = crop_mm_3D(ct_data, spacing, offset, crop_mm)
        filepath = os.path.join(dest_set.path, 'data', 'patients', p, 'study_0', 'ct', f'series_0.nii.gz')
        os.makedirs(os.path.dirname(filepath))
        save_nifti(ct_data, spacing, offset, filepath)

        # Apply crop to regions.
        region_data = pat.region_data()
        for r, d in region_data.items():
            d = crop_mm_3D(d, spacing, offset, crop_mm)
            filepath = os.path.join(dest_set.path, 'data', 'patients', p, 'study_0', 'regions', 'series_1', f'{r}.nii.gz')
            save_nifti(d, spacing, offset, filepath)

    # Indicate success.
    write_flag(dest_set, f'__CONVERTED_FROM_NRRD_{dataset}_BRAIN_CROP__')
