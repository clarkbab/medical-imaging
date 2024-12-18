from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.dicom import convert_to_nifti

dataset = 'PMCC-HN-TRAIN'
regions = 'all'
anonymise = True

convert_to_nifti(dataset, regions, anonymise)
