from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.nifti.nifti import convert_to_dicom

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    convert_ct=True,
    # pat_prefix='PMCC_ReIrrad_',
    landmarks='all',
    recreate_dataset=True,
    regions='RL:PMCC-REIRRAD',
)

convert_to_dicom(dataset, dataset, **kwargs)
