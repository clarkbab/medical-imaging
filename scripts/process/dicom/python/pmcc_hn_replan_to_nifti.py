from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.datasets.dicom import convert_to_nifti_multiple_studies
from mymi.regions import RegionNames

dataset = 'PMCC-HN-REPLAN'
regions = 'all'
anonymise = True

convert_to_nifti_multiple_studies(dataset, region=regions, anonymise=anonymise)
