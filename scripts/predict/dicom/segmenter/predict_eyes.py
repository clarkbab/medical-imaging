import numpy as np
import os
import seaborn as sns
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi import config
from mymi.datasets import DicomDataset
from mymi.datasets.dicom import ROIData, RtstructConverter
from mymi import logging
from mymi.predictions.datasets.dicom import create_all_multi_segmenter_predictions, load_multi_segmenter_prediction

# Set path to data.
# This file (__file__) runs at '_internal/predict_eyes.py'. Is it unzipped from 'base_library.zip'?
os.environ['MYMI_DATA'] = os.path.dirname(os.path.abspath(__file__))

# Create predictions.
if len(sys.argv) == 1:
    raise ValueError("Please pass dataset as argument.\n\tUsage: ./predict_eyes <dataset>")
dataset = sys.argv[1]
if len(sys.argv) == 3:
    logging.info(f"Restarting predictions from patient {sys.argv[2]}.")
    restart_pat_id = sys.argv[2]
else:
    restart_pat_id = None
regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
palette = sns.color_palette('colorblind', len(regions))
colours = [tuple((255 * np.array(c)).astype(int)) for c in palette]
model = ('segmenter', 'eyes', 'loss=-0.153282-epoch=675-step=0')
model_spacing = (1, 1, 2)
check_epochs = False
crop_mm = (330, 380, 500)
n_epochs = 1000

create_all_multi_segmenter_predictions(dataset, regions, model, model_spacing, check_epochs=check_epochs, crop_mm=crop_mm, n_epochs=n_epochs, restart_pat_id=restart_pat_id)

# Convert predictions to RTSTRUCT DICOM.
# RTSTRUCT info.
default_rt_info = {
    'label': 'PMCC-AI',
    'institution-name': 'PMCC-AI'
}

set = DicomDataset(dataset)
pat_ids = set.list_patients()
for pat_id in pat_ids:
    logging.info(f"Creating RTSTRUCT DICOM for patient {pat_id}.")

    # Get ROI IDs from DICOM dataset.
    pat = set.patient(pat_id)
    rtstruct_gt = pat.default_rtstruct.rtstruct
    info_gt = RtstructConverter.get_roi_info(rtstruct_gt)
    region_map_gt = dict((set.to_internal(data['name']), id) for id, data in info_gt.items())

    # Create RTSTRUCT.
    cts = pat.get_cts()
    rtstruct_pred = RtstructConverter.create_rtstruct(cts, default_rt_info)
    frame_of_reference_uid = rtstruct_gt.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID

    for region, colour in zip(regions, colours):
        # Load prediction.
        pred = load_multi_segmenter_prediction(dataset, pat_id, model)
            
        # Match ROI number to ground truth, otherwise assign next available integer.
        if region not in region_map_gt:
            max_roi_number = 10000
            for j in range(1, max_roi_number):  # Starting counting at 1 - this will be viewed in DICOM viewer.
                if j not in region_map_gt.values():
                    region_map_gt[region] = j
                    break
                elif j == 999:
                    raise ValueError(f"Ran out of available ROI IDs - max is {max_roi_id}.")
        roi_number = region_map_gt[region]

        # Add ROI data.
        roi_data = ROIData(
            colour=colour,
            data=pred,
            name=region,
            number=roi_number
        )
        RtstructConverter.add_roi_contour(rtstruct_pred, roi_data, cts)

        # Save pred RTSTRUCT.
        filepath = os.path.join(config.directories.predictions, 'dicom', f'{pat_id}.dcm')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rtstruct_pred.save_as(filepath)

logging.info(f"Finished creating RTSTRUCT DICOMs.")
logging.info(f"RTSTRUCT DICOMs saved to {os.path.join(config.directories.predictions, 'dicom')}.")
