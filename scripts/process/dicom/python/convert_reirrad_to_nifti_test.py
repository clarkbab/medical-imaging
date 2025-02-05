from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.datasets.dicom import convert_to_nifti
from mymi.regions import RegionNames

dataset = 'PMCC-REIRRAD'
dest_dataset = 'PMCC-REIRRAD-TEST'

convert_to_nifti(dataset, dest_dataset=dest_dataset)
