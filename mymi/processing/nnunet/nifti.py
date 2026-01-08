import os
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi import logging
from mymi.processing import one_hot_encode
from mymi.transforms import crop, resample
from mymi.typing import *
from mymi.utils import *

def convert_predictions_to_nifti_single_region(
    dataset: str,
    dataset: int,
    region: Region,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log('Converting from nnU-Net single-region predictions to NIFTI', ('dataset',), (dataset,))

    # Load predictions.
    set = NiftiDataset(dataset)
    basepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/Dataset{dataset_id}/single-region/{region}"
    files = list(sorted(os.listdir(basepath)))
    for f in tqdm(files):
        if not f.endswith('.nii.gz'):
            continue
        pat_id = f.replace('.nii.gz', '')

        # Get original spacing.
        pat = set.patient(pat_id)
        orig_size = pat.ct_size
        orig_spacing = pat.ct_spacing
        orig_origin = pat.ct_origin

        # Load predicted label.
        filepath = os.path.join(basepath, f"{pat_id}.nii.gz")
        label, spacing, _ = load_nifti(filepath)
        label = one_hot_encode(label, dims=2)

        # Resample label to original spacing.
        label = resample(label, spacing=spacing, output_spacing=orig_spacing) 

        # Crop to original shape - rounding errors during resampling.
        crop_box = ((0, 0, 0), orig_size)
        label = crop(label, crop_box)

        # Save image.
        filepath = os.path.join(set.path, 'data', 'predictions', pat_id, 'study_0', 'regions', 'series_1', region, 'nnunet.nii.gz')
        label = label.argmax(0).astype(np.bool_)
        save_nifti(label, filepath, spacing=orig_spacing, origin=orig_origin)

def convert_predictions_to_nifti_multi_region(
    dataset: str,
    dataset: int,
    regions: Regions,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log('Converting from nnU-Net single-region predictions to NIFTI', ('dataset',), (dataset,))

    # Load predictions.
    set = NiftiDataset(dataset)
    basepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/Dataset{dataset_id}"
    files = list(sorted(os.listdir(basepath)))
    pat_ids = [f.replace('.nii.gz', '') for f in files if f.endswith('.nii.gz')]
    for p in tqdm(pat_ids):
        # Get original spacing.
        pat = set.patient(p)
        orig_size = pat.ct_size
        orig_spacing = pat.ct_spacing
        orig_origin = pat.ct_origin

        # Load predicted label.
        filepath = os.path.join(basepath, f"{p}.nii.gz")
        label, spacing, _ = load_nifti(filepath)
        n_channels = len(regions) + 1
        label = one_hot_encode(label, dims=n_channels)

        # Resample label to original spacing.
        label = resample(label, spacing=spacing, output_spacing=orig_spacing) 

        # Crop to original shape - rounding errors during resampling.
        crop_box = ((0, 0, 0), orig_size)
        label = crop(label, crop_box)

        # Save images.
        for i, r in enumerate(regions):
            filepath = os.path.join(set.path, 'data', 'predictions', p, 'study_0', 'regions', 'series_1', r, 'nnunet-multi.nii.gz')
            channel = i + 1 
            rlabel = label[channel].astype(np.bool_)
            save_nifti(rlabel, filepath, spacing=orig_spacing, origin=orig_origin)
