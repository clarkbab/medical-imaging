import os
import torch
from tqdm import tqdm
from typing import Optional

from mymi import dataset as ds
from mymi.dataset.raw import recreate
from mymi.dataset.raw.dicom import ROIData, RTSTRUCTConverter
from mymi.regions import to_255, RegionColours
from mymi import utils

from ...two_stage import get_patient_segmentation

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
