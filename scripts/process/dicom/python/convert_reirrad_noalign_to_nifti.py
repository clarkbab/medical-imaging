from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.datasets.dicom import convert_to_nifti
from mymi.regions import RegionNames

dataset = 'PMCC-REIRRAD-NOALIGN'
regions = 'RL:PMCC-REIRRAD'

convert_to_nifti(dataset, anonymise_patients=False, regions=regions)
