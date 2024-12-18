import fire
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.dataset import NiftiDataset
from mymi.geometry import get_extent
from mymi import logging
from mymi.postprocessing import one_hot_encode
from mymi.prediction.dataset.nifti.nifti import load_localiser_prediction
from mymi.processing.dataset.nifti.registration import load_patient_registration
from mymi.transforms import resample, resample_multi_channel, crop_4D, pad_4D

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

def convert_predictions(
    dataset: str,
    fold: int) -> None:
    logging.arg_log('Converting nnU-Net predictions', ('dataset', 'fold'), (dataset, fold))
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()
    basepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/fold-{fold}"
    files = list(sorted(os.listdir(basepath)))
    for f in tqdm(files):
        f_pat_id = f.replace('.nii.gz', '')
        logging.info(f_pat_id)
        if f_pat_id not in pat_ids:
            continue
        logging.info(f_pat_id)

        spacing = (1, 1, 2)
        if '-0' in f_pat_id:
            # Load registered data for pre-treatment scan.
            pat_id_mt = f_pat_id.replace('-0', '-1')
            pat = set.patient(pat_id_mt)
            ct_data, _ = load_patient_registration(dataset, pat_id_mt, f_pat_id, region=None)
        else:
            pat = set.patient(f_pat_id)
            ct_data = pat.ct_data
        orig_shape = ct_data.shape
        orig_spacing = pat.ct_spacing
        orig_affine = pat.ct_affine
        logging.info(f"orig: {orig_shape, orig_spacing}")

        # Get resampled shape.
        res_ct = resample(ct_data, spacing=orig_spacing, output_spacing=spacing) 
        input_shape_before_crop = res_ct.shape
        logging.info(f"resampled: {input_shape_before_crop}")

        # Load prediction and process to original size/space.
        filepath = os.path.join(basepath, f)
        data = nib.load(filepath).get_fdata()
        data = one_hot_encode(data)
        logging.info(f"encoded pred: {data.shape}")

        # Reverse 'brain' cropping.
        crop = get_brain_crop(dataset, f_pat_id, input_shape_before_crop)
        pad = get_brain_pad(input_shape_before_crop, crop)
        data = pad_4D(data, pad)
        logging.info(f"uncropped pred: {data.shape}")

        # Resample to original spacing.
        data = resample_multi_channel(data, spacing=spacing, output_spacing=orig_spacing)
        logging.info(f"resampled pred: {data.shape}")

        # Crop to original shape - rounding errors during resampling.
        crop = ((0, 0, 0), orig_shape)
        data = crop_4D(data, crop)
        logging.info(f"cropped (orig) pred: {data.shape}")

        # Save image.
        filepath = os.path.join(basepath, f"{f_pat_id}_processed.nii.gz")
        img = Nifti1Image(data.astype(np.int32), orig_affine)
        nib.save(img, filepath)

fire.Fire(convert_predictions)
