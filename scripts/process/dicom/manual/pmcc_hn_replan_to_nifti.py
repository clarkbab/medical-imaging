from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.dicom import convert_to_nifti_multiple_studies
from mymi.regions import RegionNames

dataset = 'PMCC-HN-REPLAN-TEST'
dicom_dataset = 'PMCC-HN-REPLAN'
regions = 'all'
anonymise = True

convert_to_nifti_multiple_studies(dataset, dicom_dataset=dicom_dataset, region=regions, anonymise=anonymise)
