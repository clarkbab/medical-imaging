import fire
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import sys
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.geometry import get_extent
from mymi import logging
from mymi.postprocessing import one_hot_encode
from mymi.predictions.datasets.nifti.segmentation.segmentation import load_localiser_prediction
# from mymi.processing.dataset.nifti.registration import load_patient_registration
from mymi.transforms import resample, pad

def get_brain_crop(dataset, pat_id, size) -> tuple:
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    input_spacing = pat.ct_spacing
    spacing = (1, 1, 2)
    localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
    crop_mm = (330, 380, 500)
    # Convert to voxel crop.
    crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))

    # Get brain extent.
    # Use mid-treatment brain for both mid/pre-treatment scans as this should align with registered pre-treatment brain.
    localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
    mt_pat_id = pat_id.replace('-0', '-1') if '-0' in pat_id else pat_id
    brain_label = load_localiser_prediction(dataset, mt_pat_id, localiser)
    if spacing is not None:
        brain_label = resample(brain_label, spacing=input_spacing, output_spacing=spacing)
    brain_extent = get_extent(brain_label)

    # Get crop coordinates.
    # Crop origin is centre-of-extent in x/y, and max-extent in z.
    # Cropping boundary extends from origin equally in +/- directions for x/y, and extends
    # in - direction for z.
    p_above_brain = 0.04
    crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, brain_extent[1][2])
    crop = (
        (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - int(crop_voxels[2] * (1 - p_above_brain)))),
        (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2] + int(crop_voxels[2] * p_above_brain)))
    )
    # Threshold crop values.
    min, max = crop
    min = tuple((np.max((m, 0)) for m in min))
    max = tuple((np.min((m, s)) for m, s in zip(max, size)))
    crop = (min, max)
    
    return crop

def get_brain_pad(size, crop) -> tuple:
    min, max = crop
    min = tuple(-np.array(min))
    max = tuple(np.array(size) + np.array(min))
    # Threshold pad values.
    min = tuple((np.min((m, 0)) for m in min))
    pad = (min, max)
    return pad
