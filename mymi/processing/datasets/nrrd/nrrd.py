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
from mymi.loaders import Loader
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.predictions.datasets.nrrd import create_localiser_prediction, create_segmenter_prediction, load_localiser_prediction, load_segmenter_predictions
from mymi.regions import RegionColours, RegionList, RegionNames, to_255
from mymi.regions import regions_to_list
from mymi.reporting.loaders import load_loader_manifest
from mymi.transforms import crop, resample, top_crop_or_pad
from mymi import typing
from mymi.utils import append_row, arg_to_list, load_csv, save_files_csv

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
    crop_df = load_csv('adaptive-models', 'data', f'extrema-MICCAI-2015.csv')
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
    
def convert_to_training(
    dataset: str,
    regions: typing.Regions,
    create_data: bool = True,
    dilate_iter: int = 5,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    output_size: Optional[typing.ImageSize3D] = None,
    output_spacing: Optional[typing.ImageSpacing3D] = None,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    training_dataset: Optional[str] = None) -> None:
    # Get regions.
    set = NrrdDataset(dataset)
    if regions is None:
        regions = set.list_regions()
    else:
        regions = regions_to_list(regions)

    logging.arg_log('Converting to training', ('dataset', 'regions', 'training_dataset'), (dataset, regions, training_dataset))

    # Create the dataset.
    dest_dataset = dataset if training_dataset is None else training_dataset
    if exists(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate_training(dest_dataset)
        else:
            created = False
            set_t = TrainingDataset(dest_dataset)
            __destroy_flag(set_t, '__CONVERT_FROM_NRRD_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create_training(dest_dataset)
    write_flag(set_t, '__CONVERT_FROM_NRRD_START__')

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'output-size': [str(output_size)] if output_size is not None else ['None'],
            'output-spacing': [str(output_spacing)] if output_spacing is not None else ['None'],
            'regions': [str(regions)],
        })
        params_df.to_csv(filepath, index=False)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'output-size': [str(output_size)] if output_size is not None else ['None'],
                'output-spacing': [str(output_spacing)] if output_spacing is not None else ['None'],
                'regions': [str(regions)],
            })
            params_df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(regions=regions)

    # Get exclusions.
    exc_df = set.excluded_labels

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'group-id': float,      # Can contain 'nan' values.
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())

    # Load patient grouping if present.
    group_df = set.group_index

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            logging.info(f"Processing patient '{pat_id}'.")
            # Load input data.
            patient = set.patient(pat_id)
            spacing = patient.ct_spacing
            input = patient.ct_data

            # Resample input.
            if output_spacing:
                input = resample(input, spacing=spacing, output_spacing=output_spacing)

            # Crop/pad.
            if output_size:
                # Log warning if we're cropping the FOV as we're losing information.
                if log_warnings:
                    if output_spacing:
                        fov_spacing = output_spacing
                    else:
                        fov_spacing = spacing
                    fov = np.array(input.shape) * fov_spacing
                    new_fov = np.array(output_size) * fov_spacing
                    for axis in range(len(output_size)):
                        if fov[axis] > new_fov[axis]:
                            logging.warning(f"Patient '{patient}' had FOV '{fov}', larger than new FOV after crop/pad '{new_fov}' for axis '{axis}'.")

                # Perform crop/pad.
                input = top_crop_or_pad(input, output_size, fill=input.min())

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not patient.has_regions(region):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = patient.region_data(regions=region)[region]

                # Resample data.
                if output_spacing:
                    label = resample(label, spacing=spacing, output_spacing=output_spacing)

                # Crop/pad.
                if output_size:
                    label = top_crop_or_pad(label, output_size)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    empty = False
                    __create_training_label(set_t, i, region, label)
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
    write_flag(set_t, '__CONVERT_FROM_NRRD_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

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
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')
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

def convert_segmenter_predictions_to_dicom_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: typing.ModelName,
    segmenter: typing.ModelName,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_manifest: bool = False,
    use_model_manifest: bool = False) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser, use_manifest=use_model_manifest)
    segmenter = replace_ckpt_alias(segmenter, use_manifest=use_model_manifest)
    logging.info(f"Converting segmenter predictions to DICOM for '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

    # Build test loader.
    if use_loader_manifest:
        man_df = load_loader_manifest(datasets, region, n_folds=n_folds, test_fold=test_fold)
        samples = man_df[['dataset', 'patient-id']].to_numpy()
    else:
        _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)
        test_dataset = test_loader.dataset
        samples = [test_dataset.__get_item(i) for i in range(len(test_dataset))]

    # RTSTRUCT info.
    default_rt_info = {
        'label': 'PMCC-AI-HN',
        'institution-name': 'PMCC-AI-HN'
    }

    # Create prediction RTSTRUCTs.
    for dataset, pat_id_nrrd in tqdm(samples):
        # Get ROI ID from DICOM dataset.
        nrrd_set = NrrdDataset(dataset)
        pat_id_dicom = nrrd_set.patient(pat_id_nrrd).patient_id
        set_dicom = DicomDataset(dataset)
        patient_dicom = set_dicom.patient(pat_id_dicom)
        rtstruct_gt = patient_dicom.default_rtstruct.rtstruct
        info_gt = RtStructConverter.get_roi_info(rtstruct_gt)
        region_map_gt = dict((set_dicom.to_internal(data['name']), id) for id, data in info_gt.items())

        # Create RTSTRUCT.
        cts = patient_dicom.get_cts()
        rtstruct_pred = RtStructConverter.create_rtstruct(cts, default_rt_info)

        # Load prediction.
        pred = load_patient_segmenter_prediction(dataset, pat_id_nrrd, localiser, segmenter)
        
        # Add ROI.
        roi_data = ROIData(
            colour=list(to_255(getattr(RegionColours, region))),
            data=pred,
            name=region,
            number=region_map_gt[region]        # Patient should always have region (right?) - we created the loaders based on patient regions.
        )
        RtStructConverter.add_roi_contour(rtstruct_pred, roi_data, cts)

        # Save prediction.
        # Get localiser checkpoint and raise error if multiple.
        # Hack - clean up when/if path limits are removed.
        if config.environ('PETER_MAC_HACK') == 'True':
            base_path = 'S:\\ImageStore\\HN_AI_Contourer\\short\\dicom'
            if dataset == 'PMCC-HN-TEST':
                pred_path = os.path.join(base_path, 'test')
            elif dataset == 'PMCC-HN-TRAIN':
                pred_path = os.path.join(base_path, 'train')
        else:
            pred_path = os.path.join(nrrd_set.path, 'predictions', 'segmenter')
        filepath = os.path.join(pred_path, *localiser, *segmenter, f'{pat_id_dicom}.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct_pred.save_as(filepath)

def combine_segmenter_predictions_from_all_patients(
    dataset: Union[str, List[str]],
    n_pats: int,
    model_type: str = 'clinical') -> None:
    datasets = arg_to_list(dataset, str)
    logging.arg_log("Combining (NRRD) segmenter predictions from 'all-patients.csv'", ('dataset', 'n_pats', 'model_type'), (datasets, n_pats, model_type))

    # Load 'all-patients.csv'.
    df = load_csv('transfer-learning', 'data', 'all-patients.csv')
    df = df.astype({ 'patient-id': str })
    df = df.head(n_pats)

    cols = {
        'region': str,
        'model': str
    }

    for _, (dataset, pat_id) in tqdm(df.iterrows()):
        index_df = pd.DataFrame(columns=cols.keys())

        for region in RegionNames:
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')

            # Find fold that didn't use this patient for training.
            for test_fold in range(5):
                man_df = load_loader_manifest(datasets, region, test_fold=test_fold)
                man_df = man_df[(man_df.loader == 'test') & (man_df['origin-dataset'] == dataset) & (man_df['origin-patient-id'] == pat_id)]
                if len(man_df) == 1:
                    break
            
            # Select segmenter that didn't use this patient for training.
            if len(man_df) == 1:
                # Patient was excluded when training model for 'test_fold'.
                segmenter = (f'segmenter-{region}-v2', f'{model_type}-fold-{test_fold}-samples-None', 'best')
            elif len(man_df) == 0:
                # This patient region wasn't used for training any models, let's just use the model of the first fold.
                segmenter = (f'segmenter-{region}-v2', f'{model_type}-fold-0-samples-None', 'best') 
            else:
                raise ValueError(f"Found multiple matches in loader manifest for test fold '{test_fold}', dataset '{dataset}', patient '{pat_id}' and region '{region}'.")

            # Add index row.
            data = {
                'region': region,
                'model': f'{model_type}-fold-{test_fold}-samples-None'
            }
            index_df = append_row(index_df, data)

            # Load/create segmenter prediction.
            try:
                pred = load_segmenter_predictions(dataset, pat_id, localiser, segmenter)
            except ValueError as e:
                logging.info(str(e))
                create_localiser_prediction(dataset, pat_id, localiser)
                create_segmenter_prediction(dataset, pat_id, localiser, segmenter)
                pred = load_segmenter_predictions(dataset, pat_id, localiser, segmenter)

            # Copy prediction to new location.
            filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nrrd', dataset, pat_id, f'{region}.npz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=pred)

        # Save patient index.
        filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'predictions', 'nrrd', dataset, pat_id, 'index.csv')
        index_df.to_csv(filepath, index=False)
