import numpy as np
import nrrd
import os
import pandas as pd
import pydicom as dcm
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import List, Optional, Union

from mymi import config
from mymi import datasets as ds
from mymi.datasets.dicom import DicomDataset, ROIData, RtStructConverter
from mymi.datasets.nrrd import NrrdDataset
from mymi.datasets.nrrd import recreate as recreate_nrrd
from mymi.datasets.training import TrainingDataset, exists
from mymi.datasets.training import create as create_training
from mymi.datasets.training import recreate as recreate_training
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.predictions.datasets.nrrd import create_localiser_prediction, create_segmenter_prediction, load_localiser_prediction, load_segmenter_predictions
from mymi.regions import RegionColours, RegionList, RegionNames, to_255
from mymi.regions import regions_to_list
from mymi.transforms import crop, resample
from mymi import typing
from mymi.utils import append_row, arg_to_list, load_files_csv, save_files_csv

from ...processing import convert_to_dicom as convert_to_dicom_base, write_flag

def convert_brain_crop_to_training(
    dataset: str,
    **kwargs) -> None:
    set = NrrdDataset(dataset)
    convert_brain_crop_to_training_base(set, load_localiser_prediction=load_localiser_prediction, **kwargs)

def convert_miccai_2015_to_manual_crop_training(crop_margin: float = 10) -> None:
    dataset = 'MICCAI-2015'
    dest_dataset = f'MICCAI-2015-MC-{crop_margin}'
    regions = list(RegionList.MICCAI)
    spacing = (1, 1, 2)
    logging.info(f'Converting NRRD:{dataset} dataset to TRAINING. Regions: {regions}')

    # Create the dataset.
    if exists(dest_dataset):
        set_t = recreate_training(dest_dataset)
    else:
        set_t = create_training(dest_dataset)
    write_flag(set_t, f'__CONVERT_FROM_NRRD_START__')

    # Write params.
    filepath = os.path.join(set_t.path, 'params.csv')
    params_df = pd.DataFrame({
        'regions': [str(regions)],
        'spacing': [str(spacing)],
    })
    params_df.to_csv(filepath, index=False)

    # Load patients.
    set = NrrdDataset(dataset)
    pat_ids = set.list_patients()

    # Load crop file.
    crop_df = load_files_csv('adaptive-models', 'data', f'extrema-MICCAI-2015.csv')
    crop_df = crop_df.pivot(index='patient-id', columns='axis', values=['min-voxel', 'max-voxel'])

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())
    index = index.astype(cols)

    # Write each patient to dataset.
    start = time()
    for i, pat_id in enumerate(tqdm(pat_ids)):
        # Load input data.
        patient = set.patient(pat_id)
        input_spacing = patient.ct_spacing
        input = patient.ct_data

        # Crop input.
        # Perform before resampling as AnatomyNet crop values use the original spacing.
        pat_crop = crop_df.loc[pat_id]
        pat_crop_margin_voxel = crop_margin / np.array(input_spacing)
        crop = (
            tuple((pat_crop['min-voxel'] - pat_crop_margin_voxel).astype(int)),
            tuple((pat_crop['max-voxel'] + pat_crop_margin_voxel).astype(int)),
        )
        logging.info(pat_id)
        logging.info(crop)
        input = crop(input, crop)

        # Resample input.
        if spacing is not None:
            input = resample(input, spacing=input_spacing, output_spacing=spacing)

        # Save input.
        __create_training_input(set_t, i, input)

        for region in regions:
            # Skip if patient doesn't have region.
            if not set.patient(pat_id).has_regions(region):
                continue

            # Load label data.
            label = patient.region_data(region=region)[region]

            # Crop label.
            # Perform before resampling as AnatomyNet crop values use the original spacing.
            label = crop(label, crop)

            # Resample data.
            label = resample(label, spacing=input_spacing, output_spacing=spacing)

            # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
            if label.sum() != 0:
                empty = False
                __create_training_label(set_t, i, region, label)
            else:
                empty = True

            # Add index entry.
            data = {
                'dataset': set_t.name,
                'sample-id': i,
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
    write_flag(set_t, f'__CONVERT_FROM_{set.type.name}_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

def convert_to_dicom(
    dataset: str,
    dest_dataset: str,
    **kwargs) -> None:
    convert_to_dicom_base(NrrdDataset(dataset), dest_dataset, **kwargs)

def create_excluded_brainstem(
    dataset: str,
    dest_dataset: str) -> None:
    # Copy dataset to destination.
    set = NrrdDataset(dataset)
    dest_set = recreate_nrrd(dest_dataset)
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
        if not pat.has_regions('Brainstem'):
            continue

        # Load label data.
        data = pat.region_data(region=['Brain', 'Brainstem'])

        # Perform exclusion.
        brain_data = data['Brain'] & ~data['Brainstem']

        # Write new label.
        ct_spacing = pat.ct_spacing
        ct_offset = pat.ct_offset
        affine_spacing = np.array([
            [ct_spacing[0], 0, 0],
            [0, ct_spacing[1], 0],
            [0, 0, ct_spacing[2]]
        ])
        affine_offset = np.array(ct_offset)
        header = {
            'space directions': affine_spacing,
            'space origin': affine_offset
        }
        filepath = os.path.join(dest_set.path, 'data', 'regions', 'Brain', f'{pat_id}.nrrd')
        nrrd.write(filepath, brain_data, header=header)

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
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NRRD_TIME_HOURS_{hours}__')
    Path(path).touch()

def __create_training_input(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(dataset.path, 'data', 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def __create_training_label(
    dataset: 'Dataset',
    index: int,
    region: str,
    data: np.ndarray) -> None:
    # Save the label data.
    filepath = os.path.join(dataset.path, 'data', 'labels', region, f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def convert_segmenter_predictions_to_dicom_from_all_patients(
    n_pats: int,
    anonymise: bool = True) -> None:
    logging.arg_log('Converting segmenter predictions to DICOM', ('n_pats', 'anonymise'), (n_pats, anonymise))

    # Load 'all-patients.csv'.
    df = load_files_csv('transfer-learning', 'data', 'all-patients.csv')
    df = df.astype({ 'patient-id': str })
    df = df.head(n_pats)

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI-HN',
        'institution-name': 'PMCC-AI-HN'
    }

    # Create index.
    if anonymise:
        cols = {
            'patient-id': str,
            'anon-id': str
        }
        index_df = pd.DataFrame(columns=cols.keys())

    for i, (dataset, pat_id) in tqdm(df.iterrows()):
        # Get ROI ID from DICOM dataset.
        nrrd_set = NrrdDataset(dataset)
        pat_id_dicom = nrrd_set.patient(pat_id).patient_id
        set_dicom = DicomDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.rtstruct
        info_gt = RtStructConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RtStructConverter.create_rtstruct(cts, default_rt_info)
        frame_of_reference_uid = rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

        for region in RegionNames:
            # Load prediction.
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nrrd', dataset, pat_id, f'{region}.npz')
            pred = np.load(filepath)['data']
            
            # Match ROI number to ground truth, otherwise assign next available integer.
            if region not in region_map_gt:
                for j in range(1, 1000):
                    if j not in region_map_gt.values():
                        region_map_gt[region] = j
                        break
                    elif j == 999:
                        raise ValueError(f'Unlikely')
            roi_number = region_map_gt[region]

            # Add ROI data.
            roi_data = ROIData(
                colour=list(to_255(getattr(RegionColours, region))),
                data=pred,
                name=region,
                number=roi_number
            )
            RtStructConverter.add_roi_contour(rtstruct_pred, roi_data, cts)

        # Add index row.
        if anonymise:
            anon_id = f'PMCC_AI_HN_{i + 1:03}'
            data = {
                'patient-id': pat_id,
                'anon-id': anon_id
            }
            index_df = append_row(index_df, data)

        # Save pred RTSTRUCT.
        pat_id_folder = anon_id if anonymise else pat_id_dicom
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'rtstruct', 'pred.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if anonymise:
            rtstruct_pred.PatientID = anon_id
            rtstruct_pred.PatientName = anon_id
        rtstruct_pred.save_as(filepath)

        # Copy CTs.
        for j, path in enumerate(patient_dicom.default_rtstruct.ref_ct.paths):
            ct = dcm.read_file(path)
            if anonymise:
                ct.PatientID = anon_id
                ct.PatientName = anon_id
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'ct', f'{j}.dcm')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            ct.save_as(filepath)

        # Copy ground truth RTSTRUCT.
        rtstruct_gt = patient_dicom.default_rtstruct.rtstruct
        if anonymise:
            rtstruct_gt.PatientID = anon_id
            rtstruct_gt.PatientName = anon_id
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'dicom', pat_id_folder, 'rtstruct', 'gt.dcm')
        rtstruct_gt.save_as(filepath)
    
    # Save index.
    if anonymise:
        save_files_csv(index_df, 'transfer-learning', 'data', 'predictions', 'dicom', 'index.csv')
