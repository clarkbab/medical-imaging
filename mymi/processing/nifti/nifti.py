from datetime import datetime
import json
import matplotlib
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import *

from mymi import config
from mymi import datasets as ds
from mymi.datasets import DicomDataset, NiftiDataset
from mymi.datasets.dicom import ROIData, RtStructConverter, recreate as recreate_dicom
from mymi.datasets.nifti import create_region
from mymi.datasets.training import TrainingDataset, exists as exists_training
from mymi.datasets.training import create as create_training
from mymi.datasets.training import recreate as recreate_training
from mymi.geometry import extent, fov_centre
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.processing.processing import convert_brain_crop_to_training as convert_brain_crop_to_training_base
from mymi.regions import RegionColours, RegionList, RegionNames, regions_to_list
from mymi.transforms import resample
from mymi.typing import *
from mymi.utils import *

from ...processing import write_flag

def combine_labels(
    dataset: str,
    pat_ids: PatientIDs,
    study_ids: StudyIDs,
    region_ids: RegionID,
    output_region_id: RegionID,
    data_id: NiftiSeriesID = 'series_1',
    dry_run: bool = True) -> None:
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids)
    for p in tqdm(pat_ids):
        pat = set.patient(p)
        pat_study_ids = arg_to_list(study_ids, StudyID, literals={ 'all': pat.list_studies })
        for s in tqdm(pat_study_ids, leave=False):
            study = pat.study(s)
            region_data = study.data(data_id, 'regions').data(region_ids=region_ids)
            label = np.clip(np.stack([v for v in region_data.values()]).sum(axis=0), 0, 1).astype(bool)
            create_region(dataset, p, s, data_id, output_region_id, label, study.ct_spacing, study.ct_offset, dry_run=dry_run)

def convert_replan_to_nnunet_ref_model(
    regions: Regions,
    n_regions: int,
    create_data: bool = True,
    crop: Optional[Size3D] = None,
    crop_mm: Optional[Fov3D] = None,
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

    # Use all regions if region is 'None'.
    set = NiftiDataset(dataset)
    exc_df = set.excluded_labels

    # Create the datasets.
    filepath = os.path.join(config.directories.datasets, 'nnunet', 'raw')
    regions = regions_to_list(regions)
    all_regions = regions_to_list('RL:PMCC_REPLAN')
    for r in regions:
        region_idx = all_regions.index(r)
        dataset_id = 21 + (test_fold * n_regions) + region_idx
        dest_dataset = f"Dataset{dataset_id:03}_REF_MODEL_SINGLE_REGION_{r}_FOLD_{test_fold}"
        datapath = os.path.join(filepath, dest_dataset)
        if os.path.exists(datapath):
            shutil.rmtree(datapath)
        os.makedirs(datapath)
        jsonpath = os.path.join(datapath, 'dataset.json')
        trainpath = os.path.join(datapath, 'imagesTr')
        testpath = os.path.join(datapath, 'imagesTs')
        trainpathlabels = os.path.join(datapath, 'labelsTr')
        testpathlabels = os.path.join(datapath, 'labelsTs')

        # Get reference model split.
        filepath = f"/data/gpfs/projects/punim1413/mymi/runs/segmenter-replan-112"
        files = os.listdir(filepath)
        files = [f for f in files if f.startswith(f'n-folds-5-fold-{test_fold}')]
        assert len(files) == 1
        filepath = os.path.join(filepath, files[0])
        files = list(sorted(os.listdir(filepath)))
        file = files[-1]
        filepath = os.path.join(filepath, file, 'multi-loader-manifest.csv')
        df = pd.read_csv(filepath)
        train_pat_ids = list(df[df['loader'].isin(['train', 'validate'])]['origin-patient-id'])
        train_pat_ids = [p for p in train_pat_ids if set.patient(p).has_region(r)]
        test_pat_ids = list(df[df['loader'] == 'test']['origin-patient-id'])
        test_pat_ids = [p for p in test_pat_ids if set.patient(p).has_region(r)]

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
                    brain_extent = extent(brain_label)

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
                    input = crop(input, crop)

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
                if not set.patient(pat_id).has_region(r):
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
                    label = crop(label, crop)

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

def convert_replan_to_training(
    dataset: str,
    create_data: bool = True,
    crop: Optional[Size3D] = None,
    crop_mm: Optional[Fov3D] = None,
    dest_dataset: Optional[str] = None,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    recreate_dataset: bool = True,
    region: Optional[Regions] = None,
    round_dp: Optional[int] = None,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log('Converting NIFTI dataset to TRAINING', ('dataset', 'region'), (dataset, region))
    regions = regions_to_list(region)

    # Use all regions if region is 'None'.
    set = NiftiDataset(dataset)
    if regions is None:
        regions = set.list_regions()

    # Create the dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists_training(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate_training(dest_dataset)
        else:
            created = False
            set_t = TrainingDataset(dest_dataset)
            __destroy_flag(set_t, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create_training(dest_dataset)
    write_flag(set_t, '__CONVERT_FROM_NIFTI_START__')

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'crop': [str(crop)] if crop is not None else ['None'],
            'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath, index=False)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'crop': [str(crop)] if crop is not None else ['None'],
                'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
                'regions': [str(regions)],
            })
            params_df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(region=regions)

    # Get exclusions.
    exc_df = set.excluded_labels

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'group-id': float,
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())
    index = index.astype(cols)

    # Load patient grouping if present.
    group_df = set.group_index

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            # Load input data.
            pat = set.patient(pat_id)
            if '-0' in pat_id:
                # Load registered data for pre-treatment scan.
                pat_id_mt = pat_id.replace('-0', '-1')
                # input, region_data = load_patient_registration(dataset, pat_id_mt, pat_id, regions=region, regions_ignore_missing=True)
                input_spacing = set.patient(pat_id_mt).ct_spacing
            else:
                pat_id_mt = pat_id
                input = pat.ct_data
                input_spacing = pat.ct_spacing
                region_data = pat.region_data(regions=region, regions_ignore_missing=True) 

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
                brain_extent = extent(brain_label)

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

                # Crop input.
                input = crop(input, crop)

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_region(region):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = region_data[region]

                # Resample label.
                if spacing is not None:
                    label = resample(label, spacing=input_spacing, output_spacing=spacing)

                # Crop/pad.
                if crop_mm is not None:
                    label = crop(label, crop)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    empty = False
                    __create_training_label(set_t, i, label, region=region)
                else:
                    empty = True

                # Add index entry.
                if group_df is not None:
                    tdf = group_df[group_df['patient-id'] == pat_id]
                    if len(tdf) == 0:
                        group_id = np.nan
                    else:
                        assert len(tdf) == 1
                        group_id = tdf.iloc[0]['group-id']
                else:
                    group_id = np.nan
                data = {
                    'dataset': set_t.name,
                    'sample-id': i,
                    'group-id': group_id,
                    'origin-dataset': set.name,
                    'origin-patient-id': pat_id,
                    'region': region,
                    'empty': empty
                }
                index = append_row(index, data)

    end = time()

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Indicate success.
    write_flag(set_t, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

def convert_population_lens_crop_to_training(
    dataset: str,
    create_data: bool = True,
    crop: Optional[Size3D] = None,
    crop_method: Literal['low', 'uniform'] = 'low',
    crop_mm: Optional[Fov3D] = None,
    dest_dataset: Optional[str] = None,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    recreate_dataset: bool = True,
    region: Optional[Regions] = None,
    round_dp: Optional[int] = None,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log('Converting NIFTI dataset to TRAINING', ('dataset', 'region'), (dataset, region))
    regions = regions_to_list(region)

    # Use all regions if region is 'None'.
    set = NiftiDataset(dataset)
    if regions is None:
        regions = set.list_regions()

    # Create the dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists_training(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate_training(dest_dataset)
        else:
            created = False
            set_t = TrainingDataset(dest_dataset)
            __destroy_flag(set_t, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create_training(dest_dataset)
    write_flag(set_t, '__CONVERT_FROM_NIFTI_START__')

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'crop': [str(crop)] if crop is not None else ['None'],
            'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath, index=False)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'crop': [str(crop)] if crop is not None else ['None'],
                'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
                'regions': [str(regions)],
            })
            params_df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(region=regions)

    # Get exclusions.
    exc_df = set.excluded_labels

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'group-id': float,
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())
    index = index.astype(cols)

    # Load patient grouping if present.
    group_df = set.group_index

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            # Load input data.
            pat = set.patient(pat_id)
            if '-0' in pat_id:
                # Load registered data for pre-treatment scan.
                pat_id_mt = pat_id.replace('-0', '-1')
                # input, region_data = load_patient_registration(dataset, pat_id_mt, pat_id, region=regions, regions_ignore_missing=True)
                input_spacing = set.patient(pat_id_mt).ct_spacing
            else:
                pat_id_mt = pat_id
                input = pat.ct_data
                input_spacing = pat.ct_spacing
                region_data = pat.region_data(region=regions, regions_ignore_missing=True) 

            # Resample input.
            if spacing is not None:
                input = resample(input, spacing=input_spacing, output_spacing=spacing)

            # Crop input.
            if crop_mm is not None:
                # Convert to voxel crop.
                crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))

                # Get brain extent.
                localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
                brain_label = load_localiser_prediction(dataset, pat_id, localiser)
                if spacing is not None:
                    brain_label = resample(brain_label, spacing=input_spacing, output_spacing=spacing)
                brain_extent = extent(brain_label)
                
                if crop_method == 'low':
                    # Find lowest point containing eye/lens.
                    min_z = np.inf
                    min_region = None
                    regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
                    for region in regions:
                        if pat.has_region(region):
                            region_data = pat.region_data(region=region)[region]
                            region_extent = extent(region_data)
                            if region_extent[0][2] < min_z:
                                min_z = region_extent[0][2]
                                min_region = region

                    # Use brain extent for x/y and eye/lens extent for z.
                    crop_margin = 20 if 'Eye' in min_region else 30
                    crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, min_z - crop_margin)

                elif crop_method == 'uniform':
                    # Get extent centre of first available region.
                    centre_z = None
                    regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
                    for region in regions:
                        if pat.has_region(region):
                            rdata = pat.region_data(region=region)[region]
                            extent_centre = fov_centre(rdata)
                            centre_z = extent_centre[2]
                            break

                    # Draw z from uniform distribution around 'centre_z'.
                    width_z = 30
                    min_z = centre_z - width_z / 2
                    max_z = centre_z + width_z / 2
                    crop_origin_z = np.random.uniform(min_z, max_z)
                    crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, crop_origin_z)

                # Crop input.
                crop = (
                    (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - crop_voxels[2])),
                    (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2]))
                )
                input = crop(input, crop)

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_region(region):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = region_data[region]

                # Resample label.
                if spacing is not None:
                    label = resample(label, spacing=input_spacing, output_spacing=spacing)

                # Crop/pad.
                if crop_mm is not None:
                    label = crop(label, crop)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    empty = False
                    __create_training_label(set_t, i, label, region=region)
                else:
                    empty = True

                # Add index entry.
                if group_df is not None:
                    tdf = group_df[group_df['patient-id'] == pat_id]
                    if len(tdf) == 0:
                        group_id = np.nan
                    else:
                        assert len(tdf) == 1
                        group_id = tdf.iloc[0]['group-id']
                else:
                    group_id = np.nan
                data = {
                    'dataset': set_t.name,
                    'sample-id': i,
                    'group-id': group_id,
                    'origin-dataset': set.name,
                    'origin-patient-id': pat_id,
                    'region': region,
                    'empty': empty
                }
                index = append_row(index, data)

    end = time()

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Indicate success.
    write_flag(set_t, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

def convert_replan_to_nnunet_bootstrap() -> None:
    dataset = 'PMCC-HN-REPLAN'
    dest_dataset = 'PMCC-HN-REPLAN-BOOT'
    logging.arg_log('Converting NIFTI dataset to NIFTI (bootstrap)', ('dataset', 'dest_dataset'), (dataset, dest_dataset))

    # Create the dataset.
    dset = recreate_nifti(dest_dataset)
    write_flag(dset, '__CONVERT_FROM_NIFTI_START__')

    # Copy files.
    oset = NiftiDataset(dataset)
    files = ['excluded-labels.csv', 'processed-labels.csv']
    for filename in files:
        filepath = os.path.join(oset.path, filename)
        if os.path.islink(filepath):
            src = os.readlink(filepath)
            filepath = os.path.join(dset.path, filename)
            os.symlink(src, filepath)
        else:
            df = pd.read_csv(filepath)
            filepath = os.path.join(dset.path, filename)
            df.to_csv(filepath, index=False)

    # Write each patient to dataset.
    regions = oset.list_regions()
    start = time()
    for pat_id in tqdm(pat_ids):
        # Load CT data.
        pat = oset.patient(pat_id)
        data = pat.ct_data
        spacing = pat.ct_spacing

        # Create NIFTI CT image.
        offset = pat.ct_offset
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(dset.path, 'data', 'ct', f'{pat_id}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        region_data = pat.region_data()
        for region in regions:
            if region not in region_data:
                # Get 
                # Load 'RM' model prediction.
                model = ('segmenter-replan-112', 'n-folds')
            else:
                data = region_data[region]

            # Save NIFTI label.
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(dset.path, 'data', 'regions', region, f'{pat_id}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

    end = time()

    # Indicate success.
    write_flag(dset, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(dset, hours)

def convert_replan_to_lens_crop(
    dataset: str,
    pat_ids: List[PatientID],
    dest_dataset: str,
    crop_method: str,
    crop_mm: Tuple[float],
    region: Optional[Regions] = None) -> None:
    logging.arg_log('Converting NIFTI dataset to NIFTI (lens crop)', ('dataset', 'pat_ids', 'dest_dataset', 'crop_method', 'crop_mm'), (dataset, pat_ids, dest_dataset, crop_method, crop_mm))
    regions = regions_to_list(region)

    # Create the dataset.
    dset = recreate_nifti(dest_dataset)
    write_flag(dset, '__CONVERT_FROM_NIFTI_START__')

    # Copy files.
    oset = NiftiDataset(dataset)
    files = ['excluded-labels.csv', 'processed-labels.csv']
    for filename in files:
        filepath = os.path.join(oset.path, filename)
        if os.path.islink(filepath):
            src = os.readlink(filepath)
            filepath = os.path.join(dset.path, filename)
            os.symlink(src, filepath)
        else:
            df = pd.read_csv(filepath)
            filepath = os.path.join(dset.path, filename)
            df.to_csv(filepath, index=False)

    # Write each patient to dataset.
    start = time()
    for pat_id in tqdm(pat_ids):
        # Load CT data.
        pat = oset.patient(pat_id)
        data = pat.ct_data
        spacing = pat.ct_spacing

        # Crop using chosen method.
        if crop_method == 'lens-low':
            crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))

            # Get brain extent.
            localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
            brain_label = load_localiser_prediction(dataset, pat_id, localiser)
            brain_extent = extent(brain_label)
            
            # Find lowest point containing eye/lens.
            min_z = np.inf
            min_region = None
            eye_regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
            for eye_region in eye_regions:
                if pat.has_region(eye_region):
                    region_data = pat.region_data(region=eye_region)[eye_region]
                    region_extent = extent(region_data)
                    if region_extent[0][2] < min_z:
                        min_z = region_extent[0][2]
                        min_region = eye_region

            # Use brain extent for x/y and eye/lens extent for z.
            crop_margin = 20 if 'Eye' in min_region else 30
            crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, min_z - crop_margin)

            # Crop input.
            crop = (
                (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - crop_voxels[2])),
                (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2]))
            )
            data = crop(data, crop)
        
        elif crop_method == 'lens-uniform':
            # Convert to voxel crop.
            crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))

            # Get brain extent.
            localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
            brain_label = load_localiser_prediction(dataset, pat_id, localiser)
            brain_extent = extent(brain_label)

            # Get extent centre of first available region.
            centre_z = None
            eye_regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
            for eye_region in eye_regions:
                if pat.has_region(eye_region):
                    rdata = pat.region_data(region=eye_region)[eye_region]
                    extent_centre = fov_centre(rdata)
                    centre_z = extent_centre[2]
                    break

            # Draw z from uniform distribution around 'centre_z'.
            width_z = 30
            min_z = centre_z - width_z / 2
            max_z = centre_z + width_z / 2
            crop_origin_z = np.random.uniform(min_z, max_z)
            crop_origin = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, crop_origin_z)

            # Crop input.
            crop = (
                (int(crop_origin[0] - crop_voxels[0] // 2), int(crop_origin[1] - crop_voxels[1] // 2), int(crop_origin[2] - crop_voxels[2])),
                (int(np.ceil(crop_origin[0] + crop_voxels[0] / 2)), int(np.ceil(crop_origin[1] + crop_voxels[1] / 2)), int(crop_origin[2]))
            )
            data = crop(data, crop)
        
        else:
            raise ValueError(f"Unrecognised crop method '{crop_method}'.")

        # Create NIFTI CT image.
        offset = pat.ct_offset
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(dset.path, 'data', 'ct', f'{pat_id}.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        region_data = pat.region_data()
        for region, data in region_data.items():
            if region not in regions:
                continue

            # Crop data.
            data = crop(data, crop)
            
            # Save NIFTI label.
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(dset.path, 'data', 'regions', region, f'{pat_id}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

    end = time()

    # Indicate success.
    write_flag(dset, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(dset, hours)

def convert_brain_crop_to_training(
    dataset: str,
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    convert_brain_crop_to_training_base(set, load_localiser_prediction=load_localiser_prediction, **kwargs)
def create_excluded_brainstem(
    dataset: str,
    dest_dataset: str) -> None:
    # Copy dataset to destination.
    set = NiftiDataset(dataset)
    dest_set = recreate_nifti(dest_dataset)
    os.rmdir(dest_set.path)
    shutil.copytree(set.path, dest_set.path)

    cols = {
        'patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Get patient with 'Brain' label.
    pat_ids = dest_set.list_patients(region='Brain')
    for pat_id in tqdm(pat_ids):
        # Skip if no 'Brainstem'.
        pat = dest_set.patient(pat_id)
        if not pat.has_region('Brainstem'):
            continue

        # Load label data.
        data = pat.region_data(region=['Brain', 'Brainstem'])

        # Perform exclusion.
        brain_data = data['Brain'] & ~data['Brainstem']

        # Write new label.
        ct_spacing = pat.ct_spacing
        ct_offset = pat.ct_offset
        affine = np.array([
            [ct_spacing[0], 0, 0, ct_offset[0]],
            [0, ct_spacing[1], 0, ct_offset[1]],
            [0, 0, ct_spacing[2], ct_offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(brain_data.astype(np.int32), affine)
        filepath = os.path.join(dest_set.path, 'data', 'regions', 'Brain', f'{pat_id}.nii.gz')
        nib.save(img, filepath)

        # Add to index.
        data = {
            'patient-id': pat_id
        }
        df = append_row(df, data)

    # Save index.
    filepath = os.path.join(dest_set.path, 'excl-index.csv')
    df.to_csv(filepath, index=False)

def __destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def __print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NIFTI_TIME_HOURS_{hours}__')
    Path(path).touch()

def __create_training_input(
    dataset: 'Dataset',
    index: Union[int, str],
    data: np.ndarray,
    region: Optional[Region] = None,
    use_compression: bool = True) -> None:
    if region is not None:
        filepath = os.path.join(dataset.path, 'data', 'inputs', region)
    else:
        filepath = os.path.join(dataset.path, 'data', 'inputs')

    if use_compression:
        filepath = os.path.join(filepath, f'{index}.npz')
    else:
        filepath = os.path.join(filepath, f'{index}.np')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
        
    if use_compression:
        logging.info(f"Saving sample {index}, filepath: {filepath}")
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)

def __create_training_label(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray,
    region: Optional[str] = None,
    use_compression: bool = True) -> None:
    if region is not None:
        filepath = os.path.join(dataset.path, 'data', 'labels', region)
    else:
        filepath = os.path.join(dataset.path, 'data', 'labels')

    if use_compression:
        filepath = os.path.join(filepath, f'{index}.npz')
    else:
        filepath = os.path.join(filepath, f'{index}.np')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    if use_compression:
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)
