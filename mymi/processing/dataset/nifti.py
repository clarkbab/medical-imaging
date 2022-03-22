from tkinter import W
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import List, Literal, Optional, Union

from mymi import types
from mymi.dataset.dicom import DICOMDataset, ROIData, RTSTRUCTConverter
from mymi.dataset.nifti import NIFTIDataset
from mymi.dataset.training import create, exists, get, recreate
from mymi import logging
from mymi.prediction.dataset.nifti import load_patient_segmenter_prediction
from mymi.regions import RegionColours, RegionNames, to_255
from mymi.transforms import resample_3D, top_crop_or_pad_3D

def convert_to_training(
    dataset: str,
    regions: types.PatientRegions,
    dest_dataset: str,
    create_data: bool = True,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None) -> None:
    if type(regions) == str:
        if regions == 'all':
            regions = RegionNames
        else:
            regions = [regions]

    # Create the dataset.
    if exists(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate(dest_dataset)
        else:
            created = False
            set_t = get(dest_dataset)
            _destroy_flag(set_t, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create(dest_dataset)
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_START__')

    # Notify user.
    logging.info(f"Creating dataset '{set_t}' with recreate_dataset={recreate_dataset}, regions={regions}, dilate_regions={dilate_regions}, dilate_iter={dilate_iter}, size={size} and spacing={spacing}.")

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'size': [str(size)] if size is not None else ['None'],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'regions': [str(regions)],
                'size': [str(size)] if size is not None else ['None'],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
            })
            params_df.to_csv(filepath)

    # Load patients.
    set = NIFTIDataset(dataset)
    pats = set.list_patients()

    # Create index.
    cols = {
        'dataset': str,
        'patient-id': str,
        'sample-id': str
    }
    index = pd.DataFrame(columns=cols.keys())
    for i, pat in enumerate(pats):
        # Get patient regions.
        pat_regions = set.patient(pat).list_regions()
        
        # Add entries.
        for region in pat_regions:
            data = {
                'dataset': dataset,
                'patient-id': pat,
                'sample-id': i,
                'region': region
            }
            index = index.append(data, ignore_index=True)

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat in enumerate(tqdm(pats)):
            # Load input data.
            patient = set.patient(pat)
            old_spacing = patient.ct_spacing
            input = patient.ct_data

            # Resample input.
            if spacing:
                input = resample_3D(input, old_spacing, spacing)

            # Crop/pad.
            if size:
                # Log warning if we're cropping the FOV as we're losing information.
                if log_warnings:
                    if spacing:
                        fov_spacing = spacing
                    else:
                        fov_spacing = old_spacing
                    fov = np.array(input.shape) * fov_spacing
                    new_fov = np.array(size) * fov_spacing
                    for axis in range(len(size)):
                        if fov[axis] > new_fov[axis]:
                            logging.warning(f"Patient '{patient}' had FOV '{fov}', larger than new FOV after crop/pad '{new_fov}' for axis '{axis}'.")

                # Perform crop/pad.
                input = top_crop_or_pad_3D(input, size, fill=input.min())

            # Save input.
            _create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat).has_region(region):
                    continue

                # Load label data.
                label = patient.region_data(regions=region)[region]

                # Resample data.
                if spacing:
                    label = resample_3D(label, old_spacing, spacing)

                # Crop/pad.
                if size:
                    label = top_crop_or_pad_3D(label, size)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    _create_training_label(set_t, i, region, label)

    end = time()

    # Indicate success.
    _write_flag(set_t, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    _print_time(set_t, hours)

def _destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def _write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def _print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NIFTI_TIME_HOURS_{hours}__')
    Path(path).touch()

def _create_training_input(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(dataset.path, 'data', 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def _create_training_label(
    dataset: 'Dataset',
    index: int,
    region: str,
    data: np.ndarray) -> None:
    # Save the label data.
    filepath = os.path.join(dataset.path, 'data', 'labels', region, f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def convert_segmenter_predictions_to_dicom(
    dataset: str,
    regions: Union[str, List[str]],
    localiser_runs: Union[str, List[str]],
    segmenter_runs: Union[str, List[str]],
    model: str) -> None:
    if type(regions) == str:
        regions = [regions]

    # Load patients.
    set = NIFTIDataset(dataset)
    set_d = DICOMDataset(dataset)
    region_map = set_d.region_map
    def to_internal(name: str) -> str:
        if region_map is None:
            return name
        else:
            return region_map.to_internal(name)
    pats = set.list_patients()

    # Get full localiser/segmenter names.
    localisers = {}
    segmenters = {}
    for i, region in enumerate(regions):
        # Get run name for this region.
        if type(localiser_runs) == str:
            localiser_run = localiser_runs
        else:
            localiser_run = localiser_runs[i]
        if type(segmenter_runs) == str:
            segmenter_run = segmenter_runs
        else:
            segmenter_run = segmenter_runs[i]

        # Get localiser checkpoint and raise error if multiple.
        # Hack - clean up when/if path limits are removed.
        if os.environ['PETER_MAC_HACK'] == 'True':
            if dataset == 'PMCC-HN-TEST':
                pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\Test'
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = 'S:\\ImageStore\\AtlasSegmentation\\BC_HN\\Train'
        else:
            pred_path = os.path.join(set.path, 'predictions')
        loc_checks_path = os.path.join(pred_path, 'segmenter', f'localiser-{region}', localiser_run)
        loc_checks = os.listdir(loc_checks_path)
        if len(loc_checks) != 1:
            raise ValueError(f"localiser-{region} should have 1 checkpoint, got {len(loc_checks)}")
        loc_check = loc_checks[0]

        # Get segmenter checkpoint and raise error if multiple.
        loc_check_path = os.path.join(loc_checks_path, loc_check)
        seg_checks_path = os.path.join(loc_check_path, f'segmenter-{region}', segmenter_run)
        seg_checks = os.listdir(seg_checks_path)
        if len(seg_checks) != 1:
            raise ValueError(f"segmenter-{region} should have 1 checkpoint, got {len(seg_checks)}")
        seg_check = seg_checks[0]

        localisers[region] = (f'localiser-{region}', localiser_run, loc_check)
        segmenters[region] = (f'segmenter-{region}', segmenter_run, seg_check)

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI',
        'institution-name': 'PMCC-AI'
    }

    # Remove old predictions folder.
    folderpath = os.path.join(set_d.path, 'predictions', model)
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)

    for pat in tqdm(pats):
        # Get patient regions.
        patient = set.patient(pat)
        pat_regions = patient.list_regions()

        # Get DICOM RTSTRUCT - predicted ROI IDs must line up for dose grid comparisons.
        pat_d = patient.patient_id
        patient_d = set_d.patient(pat_d)
        rtstruct_d = patient_d.default_rtstruct.get_rtstruct()
        region_info_d = RTSTRUCTConverter.get_roi_info(rtstruct_d)
        region_info_d = dict((to_internal(name), id) for id, name in region_info_d)

        # Create RTSTRUCT.
        cts = patient_d.get_cts()
        rtstruct_p = RTSTRUCTConverter.create_rtstruct(cts, default_rt_info)

        for region in regions:
            # Check that patient has region.
            if region not in pat_regions:
                continue

            # Load prediction.
            try:
                pred = load_patient_segmenter_prediction(dataset, pat, localisers[region], segmenters[region])
            except ValueError as e:
                continue
        
            # Add ROI.
            roi_data = ROIData(
                colour=list(to_255(getattr(RegionColours, region))),
                data=pred,
                frame_of_reference_uid=rtstruct_d.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
                name=region,
                number=region_info_d[region]
            )
            RTSTRUCTConverter.add_roi(rtstruct_p, roi_data, cts)

        # Save prediction.
        filepath = os.path.join(folderpath, f'{pat_d}.dcm')
        folder = os.path.dirname(filepath)
        os.makedirs(folder, exist_ok=True)
        rtstruct_p.save_as(filepath)
