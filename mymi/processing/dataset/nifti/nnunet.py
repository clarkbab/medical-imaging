import json
import os
import shutil
from typing import *

from mymi import config
from mymi.dataset import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.types import *

def convert_to_nnunet_single_region(
    # Creates a single nnU-Net 'raw' dataset per region.
    dataset: str,
    first_dataset_id: int,
    spacing: ImageSpacing3D, 
    regions: PatientRegions,
    n_regions: int,
    create_data: bool = True,
    crop: Optional[ImageSize3D] = None,
    crop_mm: Optional[ImageSizeMM3D] = None,
    dest_dataset: Optional[str] = None,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    n_folds: int = 5,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    test_fold: int = 0) -> None:
    dataset = 'PMCC-HN-REPLAN'
    crop_mm = (330, 380, 500)
    spacing = (1, 1, 2)
    logging.arg_log('Converting NIFTI dataset to NNUNET (REF MODEL)', ('dataset', 'regions', 'test_fold'), (dataset, regions, test_fold))

    # Get regions.
    set = NiftiDataset(dataset)
    # We require 'all_regions' for consistent nnU-Net dataset IDs.
    all_regions = set.list_regions()
    regions = regions_to_list(regions)
    if regions is None:
        regions = set.list_regions()
    n_regions = len(regions)

    # Create the datasets.
    filepath = os.path.join(config.directories.datasets, 'nnunet', 'raw')
    for r in regions:
        # Create folders.
        region_idx = all_regions.index(r)
        dataset_id = first_dataset_id + (test_fold * n_regions) + region_idx
        dest_dataset = f"Dataset{dataset_id:03}_SINGLE_REGION_{r}_FOLD_{test_fold}"
        datapath = os.path.join(filepath, dest_dataset)
        if os.path.exists(datapath):
            shutil.rmtree(datapath)
        os.makedirs(datapath)
        trainpath = os.path.join(datapath, 'imagesTr')
        testpath = os.path.join(datapath, 'imagesTs')
        trainpathlabels = os.path.join(datapath, 'labelsTr')
        testpathlabels = os.path.join(datapath, 'labelsTs')

        # Get train/test split.

        # Create 'dataset.json'.
        dataset_json = {
            "channel_names": {
            "0": "CT"
            },
            "labels": {
            "background": 0
            },
            "numTraining": len(train_pat_ids),
            "file_ending": ".nii.gz"
        }
        for j, region in enumerate(regions):
            dataset_json["labels"][region] = j + 1
        filepath = os.path.join(datapath, 'dataset.json')
        with open(filepath, 'w') as f:
            json.dump(dataset_json, f)

        # Write training/test patients.
        pat_ids = [train_pat_ids, test_pat_ids]
        paths = [trainpath, testpath]
        labelpaths = [trainpathlabels, testpathlabels]
        for pat_ids, path, labelspath in zip(pat_ids, paths, labelpaths):
            for pat_id in tqdm(pat_ids):
                # Load input data.
                pat = set.patient(pat_id)
                if '-0' in pat_id:
                    # Load registered data for pre-treatment scan.
                    pat_id_mt = pat_id.replace('-0', '-1')
                    # input, region_data = load_patient_registration(dataset, pat_id_mt, pat_id, regions=regions, regions_ignore_missing=True)
                    input_spacing = set.patient(pat_id_mt).ct_spacing
                else:
                    pat_id_mt = pat_id
                    input = pat.ct_data
                    input_spacing = pat.ct_spacing
                    region_data = pat.region_data(regions=regions, regions_ignore_missing=True) 

                # Resample input.
                if spacing is not None:
                    input = resample(input, spacing=input_spacing, output_spacing=spacing)

                # Crop input.
                if crop_mm is not None:
                    # Convert to voxel crop.
                    crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))
                    
                    # Get brain extent.
                    # Use mid-treatment brain for both mid/pre-treatment scans as this should align with registered pre-treatment brain.
                    localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
                    brain_label = load_localiser_prediction(dataset, pat_id_mt, localiser)
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
                    max = tuple((np.min((m, s)) for m, s in zip(max, input.shape)))
                    crop = (min, max)

                    # Crop input.
                    input = crop_3D(input, crop)

                # Save data.
                filepath = os.path.join(path, f'{pat_id}_0000.nii.gz')   # '0000' indicates a single channel (CT only).
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                affine = np.array([
                    [spacing[0], 0, 0, 0],
                    [0, spacing[1], 0, 0],
                    [0, 0, spacing[2], 0],
                    [0, 0, 0, 1]])
                img = Nifti1Image(input, affine)
                nib.save(img, filepath)

                # Skip if patient doesn't have region.
                # This is a problem, as the missing label will be trained as "background".
                if not set.patient(pat_id).has_regions(r):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == r)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = region_data[r]

                # Resample label.
                if spacing is not None:
                    label = resample(label, spacing=input_spacing, output_spacing=spacing)

                # Crop/pad.
                if crop_mm is not None:
                    label = crop_3D(label, crop)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save data.
                filepath = os.path.join(labelspath, f'{pat_id}.nii.gz')   # '0000' indicates a single channel (CT only).
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                img = Nifti1Image(label.astype(np.int32), affine)
                nib.save(img, filepath)
