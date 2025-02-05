import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.processing.datasets.dicom.custom import convert_velocity_predictions_to_nifti

dataset = 'PMCC-REIRRAD'
# pat_prefix = 'PMCC_ReIrrad_'
pat_prefix = None
regions = 'RL:PMCC-REIRRAD'
transform_types = ['EDMP']

convert_velocity_predictions_to_nifti(dataset, pat_prefix=pat_prefix, regions=regions, transform_types=transform_types)
