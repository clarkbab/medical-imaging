import numpy as np
import os
import pydicom as dcm
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.dicom import DICOMDataset, ROIData, RTSTRUCTConverter
from mymi.geometry import get_extent
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.systems import Localiser, MultiSegmenter, Segmenter
from mymi.postprocessing import largest_cc_4D
from mymi.regions import to_255, RegionColours, region_to_list
from mymi.transforms import crop_3D, pad_4D, resample, resample_list
from mymi.types import Box3D, Size3D, Spacing3D, Model, ModelName, PatientID, PatientRegions
from mymi.utils import Timer, arg_broadcast, arg_to_list, encode

from ..prediction import get_localiser_prediction as get_localiser_prediction_base

def create_all_multi_segmenter_predictions(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: Union[ModelName, Model],
    model_spacing: Spacing3D,
    restart_pat_id: Optional[PatientID] = None,
    use_timing: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Making multi-segmenter predictions', ('dataset', 'region', 'model', 'model_spacing'), (dataset, region, model, model_spacing))
    datasets = arg_to_list(dataset, str)
    regions = region_to_list(region)

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Create timing table.
    if use_timing:
        cols = {
            'fold': float,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'device': str
        }
        timer = Timer(cols)

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, map_location=device, n_gpus=n_gpus, region=regions, **kwargs)

    # Load all patients.
    for dataset in tqdm(datasets):
        set = DICOMDataset(dataset)
        pat_ids = set.list_patients()

        if restart_pat_id is not None:
            idx = pat_ids.index(str(restart_pat_id))
            pat_ids = pat_ids[idx:]
        
        for pat_id in tqdm(pat_ids, leave=False):
            logging.info(f"Predicting '{dataset}:{pat_id}'.")

            # Timing table data.
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'device': device.type
            }

            with timer.record(data, enabled=use_timing):
                create_multi_segmenter_prediction(dataset, pat_id, model, regions, model_spacing, device=device, **kwargs)

    # Save timing data.
    if use_timing:
        model_name = replace_ckpt_alias(model) if type(model) == tuple else model.name
        filepath = os.path.join(config.directories.predictions, 'timing', 'multi-segmenter', encode(datasets), encode(regions), *model_name, 'timing.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def create_multi_segmenter_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[str, List[str]],
    model: Union[ModelName, Model],
    model_region: PatientRegions,
    model_spacing: Spacing3D,
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs: Dict[str, Any]) -> None:
    model_name = model if isinstance(model, tuple) else model.name
    logging.arg_log('Creating multi-segmenter prediction', ('dataset', 'pat_id', 'model', 'model_region', 'model_spacing', 'device', 'savepath'), (dataset, pat_id, model_name, model_region, model_spacing, device, savepath))
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, str)
    assert len(datasets) == len(pat_ids)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load PyTorch model.
    if type(model) == tuple:
        n_gpus = 0 if device.type == 'cpu' else 1
        model = MultiSegmenter.load(model, map_location=device, n_gpus=n_gpus, region=model_region, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Make prediction.
        pred = get_multi_segmenter_prediction(dataset, pat_id, model, model_region, model_spacing, device=device, **kwargs)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model_name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def get_multi_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    model: Union[ModelName, Model],
    model_region: PatientRegions,
    model_spacing: Spacing3D,
    device: torch.device = torch.device('cpu'),
    crop_mm: Optional[Box3D] = None,
    **kwargs) -> np.ndarray:
    model_regions = arg_to_list(model_region, str)

    # Load model.
    if type(model) == tuple:
        model = MultiSegmenter.load(*model, **kwargs)
        model.eval()
        model.to(device)

    # Load patient CT data and spacing.
    set = DICOMDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    input_spacing = patient.ct_spacing

    # Resample input to model spacing.
    input_size = input.shape
    input = resample(input, spacing=input_spacing, output_spacing=model_spacing) 
    input_size_before_crop = input.shape

    # Apply 'brain' cropping.
    assert crop_mm is not None
    if crop_mm is None:
        raise ValueError(f"'crop_mm' must be passed for 'Brain' cropping.")

    # Convert to voxel crop.
    crop_voxels = tuple((np.array(crop_mm) / np.array(model_spacing)).astype(np.int32))

    # Get brain extent.
    localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
    brain_pred_exists = load_localiser_prediction(dataset, pat_id, localiser, exists_only=True)
    if not brain_pred_exists:
        create_localiser_prediction(dataset, pat_id, localiser, check_epochs=False, device=device)
    brain_label = load_localiser_prediction(dataset, pat_id, localiser)
    brain_label = resample(brain_label, spacing=input_spacing, output_spacing=model_spacing)
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

    # Crop input.
    input = crop_3D(input, crop)

    # Pass image to model.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = model(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Apply thresholding/one-hot-encoding.
    pred = pred.argmax(dim=0)
    pred = one_hot(pred, num_classes=len(model_regions) + 1)
    pred = pred.moveaxis(-1, 0)
    pred = pred.cpu().numpy().astype(np.bool_)
    
    # Apply postprocessing.
    pred = largest_cc_4D(pred)

    # Reverse the 'brain' cropping.
    pad_min = tuple(-np.array(crop[0]))
    pad_max = tuple(np.array(pad_min) + np.array(input_size_before_crop))
    pad = (pad_min, pad_max)
    pred = pad_4D(pred, pad)

    # Resample to original spacing/size.
    pred = resample_list(pred, output_size=input_size, output_spacing=input_spacing, spacing=model_spacing)

    return pred

def load_multi_segmenter_prediction(
    dataset: str,
    pat_id: PatientID,
    model: ModelName,
    exists_only: bool = False,
    use_model_manifest: bool = False) -> Union[np.ndarray, bool]:
    pat_id = str(pat_id)
    model = replace_ckpt_alias(model, use_manifest=use_model_manifest)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'data', 'multi-segmenter', dataset, pat_id, *model, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', model '{model}'. Path: {filepath}")
    pred = np.load(filepath)['data']

    return pred

def load_localiser_prediction(
    dataset: str,
    pat_id: PatientID,
    localiser: ModelName,
    exists_only: bool = False) -> Union[np.ndarray, bool]:
    pat_id = str(pat_id)
    localiser = replace_ckpt_alias(localiser)

    # Load prediction.
    filepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser, 'pred.npz')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Prediction not found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'.")

    pred = np.load(filepath)['data']
    return pred

def create_localiser_prediction(
    dataset: Union[str, List[str]],
    pat_id: Union[PatientID, List[PatientID]],
    localiser: Union[ModelName, Model],
    device: Optional[torch.device] = None,
    savepath: Optional[str] = None,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    pat_ids = arg_to_list(pat_id, (int, str), out_type=str)
    datasets = arg_broadcast(datasets, pat_ids)
    assert len(datasets) == len(pat_ids)

    # Load gpu if available.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logging.info('Predicting on GPU...')
        else:
            device = torch.device('cpu')
            logging.info('Predicting on CPU...')

    # Load localiser.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser, map_location=device, **kwargs)

    for dataset, pat_id in zip(datasets, pat_ids):
        # Load dataset.
        set = DICOMDataset(dataset)
        pat = set.patient(pat_id)

        logging.info(f"Creating prediction for patient '{pat}', localiser '{localiser.name}'.")

        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser, device=device)

        # Save segmentation.
        if savepath is None:
            savepath = os.path.join(config.directories.predictions, 'data', 'localiser', dataset, pat_id, *localiser.name, 'pred.npz')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.savez_compressed(savepath, data=pred)

def get_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: Model,
    loc_size: Size3D = (128, 128, 150),
    loc_spacing: Spacing3D = (4, 4, 4),
    device: Optional[torch.device] = None) -> np.ndarray:
    # Load data.
    set = DICOMDataset(dataset)
    patient = set.patient(pat_id)
    input = patient.ct_data
    spacing = patient.ct_spacing

    # Make prediction.
    pred = get_localiser_prediction_base(input, spacing, localiser, loc_size=loc_size, loc_spacing=loc_spacing, device=device)

    return pred

def create_segmenter_predictions(
    dataset: str,
    localiser: ModelName,
    segmenter: ModelName,
    region: PatientRegions = 'all') -> None:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load patients.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(regions=region)

    # Load models.
    localiser_args = localiser
    segmenter_args = segmenter
    localiser = Localiser.load(*localiser)
    segmenter = Segmenter.load(*segmenter)

    # Create RTSTRUCT info.
    rt_info = {
        'label': 'MYMI',
        'institution-name': 'MYMI'
    }

    for pat in tqdm(pats):
        # Get segmentation.
        seg = get_patient_segmentation(set, pat, localiser, segmenter, device=device)

        # Load reference CT dicoms.
        cts = set.patient(pat).get_cts()

        # Create RTSTRUCT dicom.
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts, rt_info)

        # Create ROI data.
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=seg,
            frame_of_reference_uid=rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name=region
        )

        # Add ROI.
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)

        # Save in folder.
        filename = f"{pat}.dcm"
        filepath = os.path.join(set.path, 'predictions', 'two-stage', f"{localiser_args[0]}-{segmenter_args[0]}", f"{localiser_args[1]}-{segmenter_args[1]}", f"{localiser_args[2]}-{segmenter_args[2]}", filename) 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct.save_as(filepath)

def create_dataset(
    dataset: str,
    device: torch.device = torch.device('cpu'),
    output_dataset: Optional[str] = None,
    use_gpu: bool = True) -> None:
    """
    effect: generates a DICOMDataset of predictions.
    args:
        dataset: the dataset to create predictions from.
    kwargs:
        device: the device to perform inference on.
        output_dataset: the name of the dataset to hold the predictions.
        use_gpu: use GPU for matrix calculations.
    """
    # Load patients.
    source_ds = ds.get(dataset, 'dicom')
    pats = source_ds.list_patients()

    # Re/create pred dataset.
    pred_ds_name = output_dataset if output_dataset else f"{dataset}-pred"
    recreate(pred_ds_name)
    ds_pred = ds.get(pred_ds_name, type_str='dicom')

    # Create RTSTRUCT info.
    rt_info = {
        'label': 'MYMI',
        'institution-name': 'MYMI'
    }

    for pat in tqdm(pats):
        # Get segmentation.
        seg = get_patient_segmentation(source_ds, pat, device=device)

        # Load reference CT dicoms.
        cts = ds.patient(pat).get_cts()

        # Create RTSTRUCT dicom.
        rtstruct = RTSTRUCTConverter.create_rtstruct(cts, rt_info)

        # Create ROI data.
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=seg,
            frame_of_reference_uid=rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID,
            name='Parotid_L'
        )

        # Add ROI.
        RTSTRUCTConverter.add_roi(rtstruct, roi_data, cts)

        # Save in new 'pred' dataset.
        filename = f"{pat}.dcm"
        filepath = os.path.join(ds_pred.path, 'raw', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct.save_as(filepath)

def load_segmenter_predictions(
    dataset: str,
    pat_id: str,
    model: str,
    regions: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
    if type(regions) == str:
        regions = [regions]

    # Load ref CTs.
    set = ds.get(dataset, 'dicom')
    region_map = set.region_map
    patient = set.patient(pat_id)
    ref_cts = patient.get_cts()

    # Get region info.
    filepath = os.path.join(set.path, 'predictions', model, f'{pat_id}.dcm')
    rtstruct = dcm.read_file(filepath)
    region_names = RTSTRUCTConverter.get_roi_names(rtstruct)
    def to_internal(name):
        if region_map is None:
            return name
        else:
            return region_map.to_internal(name)
    name_map = dict((to_internal(name), name) for name in region_names)

    # Extract data.
    preds = []
    for region in regions:
        pred = RTSTRUCTConverter.get_roi_data(rtstruct, name_map[region], ref_cts)
        preds.append(pred)
    
    # Determine return type.
    if len(preds) == 1:
        return preds[0]
    else:
        return preds
