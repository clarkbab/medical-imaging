import logging
import os
import torch
from tqdm import tqdm
from typing import Optional

from mymi import checkpoint
from mymi import dataset as ds
from mymi.dataset.dicom import DicomDataset, ROIData, RTSTRUCTConverter
from mymi.models import SingleChannelUNet
from mymi.regions import to_255, RegionColours
from mymi import types
from mymi import utils

from .two_stage import get_patient_two_stage_segmentation

def generate_dataset_predictions(
    dataset: str,
    localiser: str,
    localiser_run: str,
    localiser_size: str,
    localiser_spacing: str,
    segmenter: str,
    segmenter_run: str,
    segmenter_size: str,
    segmenter_spacing: str,
    clear_cache: bool = False,
    localiser_checkpoint: str = 'checkpoint',
    log_level: str = 'info',
    output_dataset: Optional[str] = None,
    rtstruct_label: str = 'MYMI',
    rtstruct_institution: str = 'MYMI',
    segmenter_checkpoint: str = 'checkpoint',
    use_gpu: bool = True,
    use_postprocessing: bool = True) -> None:
    """
    effect: generates a dataset of predictions.
    args:
        dataset: the dataset to create predictions from.
        localiser: the name of the localiser model.
        localiser_run: the name of the localiser training run.
        localiser_size: the input size expected by the localiser.
        localiser_spacing: the input spacing expected by the localiser.
        segmenter: the name of the segmenter model.
        segmenter_run: the name of the segmenter training run.
        segmenter_size: the input size expected by the segmenter.
        segmenter_spacing: the input spacing expected by the segmenter.
    kwargs:
        clear_cache: force the cache to clear.
        localiser_checkpoint: the name of the localiser training run checkpoint.
        log_level: the log level.
        output_dataset: the name of the dataset to hold the predictions.
        segmenter_checkpoint: the name of the segmenter training run checkpoint.
        use_gpu: use GPU for matrix calculations.
        use_postprocessing: apply postprocessing to network predictions.
    """
    # Configure logging.
    log_level = getattr(logging, log_level.upper(), None)
    utils.configure_logging(log_level)

    # Configure device.
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
            logging.info('CUDA not available.')
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}.")

    # Load localiser model.
    loc_model = SingleChannelUNet()
    loc_model = loc_model.to(device)
    logging.info(f"Using localiser model arch: {loc_model.__class__}.")

    # Load localiser params.
    localiser_state, _ = checkpoint.load(localiser, localiser_run, checkpoint_name=localiser_checkpoint, device=device)
    loc_model.load_state_dict(localiser_state)
    logging.info(f"Loaded localiser with name '{localiser}' from training run '{localiser_run}'.")

    # Load segmenter model.
    seg_model = SingleChannelUNet()
    seg_model = seg_model.to(device)
    logging.info(f"Using segmenter model arch: {seg_model.__class__}.")

    # Load segmenter params.
    segmenter_state, _ = checkpoint.load(segmenter, segmenter_run, checkpoint_name=segmenter_checkpoint, device=device)
    seg_model.load_state_dict(segmenter_state)
    logging.info(f"Loaded segmenter with name '{segmenter}' from training run '{segmenter_run}'.")

    # Load patient IDs.
    ds.select(dataset)
    pats = ds.list_patients()

    # Re/create pred dataset.
    pred_ds_name = output_dataset if output_dataset else f"{dataset}-pred"
    dss = ds.list()
    if pred_ds_name in dss:
        ds.destroy(pred_ds_name)
    ds.create(pred_ds_name)
    ds_pred = DicomDataset(pred_ds_name)

    # Create RTSTRUCT info.
    rt_info = {
        'label': rtstruct_label,
        'institution-name': rtstruct_institution
    }

    for pat in tqdm(pats):
        # Get segmentation.
        seg = get_patient_two_stage_segmentation(pat, loc_model, localiser_size, localiser_spacing, seg_model, segmenter_size, segmenter_spacing, clear_cache=clear_cache, device=device, use_postprocessing=use_postprocessing)

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
