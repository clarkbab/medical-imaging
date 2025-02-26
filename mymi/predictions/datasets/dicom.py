import numpy as np
import os
import pydicom as dcm
import pytorch_lightning as pl
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from mymi import config
from mymi import datasets as ds
from mymi.datasets.dicom import DicomDataset, ROIData, RtstructConverter
from mymi.geometry import extent
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.lightning_modules import MultiSegmenter, Segmenter
from mymi.postprocessing import largest_cc_4D
from mymi.regions import to_255, RegionColours, regions_to_list
from mymi.transforms import crop, pad, resample
from mymi.typing import Box3D, ImageSize3D, ImageSpacing3D, ModelName, PatientID, PatientRegions
from mymi.utils import Timer, arg_broadcast, arg_to_list, encode

def create_all_multi_segmenter_predictions(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: Union[ModelName, pl.LightningModule],
    model_spacing: ImageSpacing3D,
    restart_pat_id: Optional[PatientID] = None,
    use_timing: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    logging.arg_log('Making multi-segmenter predictions', ('dataset', 'region', 'model', 'model_spacing'), (dataset, region, model, model_spacing))
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)

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
        set = DicomDataset(dataset)
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

def create_dataset(
    dataset: str,
    device: torch.device = torch.device('cpu'),
    output_dataset: Optional[str] = None,
    use_gpu: bool = True) -> None:
    """
    effect: generates a DicomDataset of predictions.
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
        rtstruct = RtstructConverter.create_rtstruct(cts, rt_info)

        # Create ROI data.
        roi_data = ROIData(
            colour=list(to_255(RegionColours.Parotid_L)),
            data=seg,
            name='Parotid_L'
        )

        # Add ROI.
        RtstructConverter.add_roi_contour(rtstruct, roi_data, cts)

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
    region_names = RtstructConverter.get_roi_names(rtstruct)
    def to_internal(name):
        if region_map is None:
            return name
        else:
            return region_map.to_internal(name)
    name_map = dict((to_internal(name), name) for name in region_names)

    # Extract data.
    preds = []
    for region in regions:
        pred = RtstructConverter.get_roi_contour(rtstruct, name_map[region], ref_cts)
        preds.append(pred)
    
    # Determine return type.
    if len(preds) == 1:
        return preds[0]
    else:
        return preds
