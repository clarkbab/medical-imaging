import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Optional, Tuple

from mymi import dataset as ds
from mymi.dataset.raw import recreate
from mymi.dataset.raw.dicom import ROIData, RTSTRUCTConverter
from mymi import logging
from mymi.models.systems import Localiser, Segmenter
from mymi.regions import to_255, RegionColours
from mymi import utils
from mymi import types

from ..two_stage import get_localiser_prediction, get_two_stage_prediction

def create_localiser_predictions(
    dataset: str,
    localiser: Tuple[str, str, str],
    clear_cache: bool = False,
    region: types.PatientRegions = 'all') -> None:
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
    localiser = Localiser.load(*localiser)

    for pat in tqdm(pats):
        # Make prediction.
        _, data = get_localiser_prediction(set, pat, localiser, clear_cache=clear_cache, device=device, return_seg=True)

        # Save in folder.
        spacing = set.patient(pat).ct_spacing()
        affine = np.ndarray([
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 0],
            [0, 0, 0, 1]
        ])
        img = nib.Nifti1Image(data, affine)
        filepath = os.path.join(set.path, 'predictions', 'localiser', f"{localiser_args[0]}-{segmenter_args[0]}", f"{localiser_args[1]}-{segmenter_args[1]}", f"{localiser_args[2]}-{segmenter_args[2]}", f"{pat}.nii.gz") 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

def create_two_stage_predictions(
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    clear_cache: bool = False,
    region: types.PatientRegions = 'all') -> None:
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
        seg = get_patient_segmentation(set, pat, localiser, segmenter, clear_cache=clear_cache, device=device)

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
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    output_dataset: Optional[str] = None,
    use_gpu: bool = True) -> None:
    """
    effect: generates a DICOMDataset of predictions.
    args:
        dataset: the dataset to create predictions from.
    kwargs:
        clear_cache: force the cache to clear.
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
        seg = get_patient_segmentation(source_ds, pat, clear_cache=clear_cache, device=device)

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
