import os
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi import logging
from mymi.postprocessing import one_hot_encode
from mymi.transforms import crop, resample
from mymi.typing import *
from mymi.utils import load_nifti, save_as_nifti

def convert_predictions_to_nifti_single_region(
    dataset: str,
    dataset_id: int,
    region: PatientRegion,
    spacing: Optional[ImageSpacing3D] = None) -> None:
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
        orig_offset = pat.ct_offset

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
        save_as_nifti(label, orig_spacing, orig_offset, filepath)
