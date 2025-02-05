import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.nifti.nifti import convert_adaptive_brain_crop_to_training_adaptive

dataset = 'PMCC-HN-REPLAN'
resolution='222'
dest_dataset = f'PMCC-HN-REPLAN-ADPT-{resolution}'
spacing = (2, 2, 2)
crop_mm = (330, 380, 500)
regions = 'RL:PMCC_REPLAN'

convert_adaptive_brain_crop_to_training_adaptive(dataset, crop_mm=crop_mm, dest_dataset=dest_dataset, region=regions, spacing=spacing)
