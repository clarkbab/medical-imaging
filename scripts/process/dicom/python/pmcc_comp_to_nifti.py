from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.datasets.dicom import convert_to_nifti
from mymi.regions import RegionNames

dataset = 'PMCC-COMP'
regions = [
    'A_Aorta', 'A_Pulmonary', 'Bladder', 'Bone_Ilium_L', 'Bone_Ilium_R','BrachialPlex_L', 'BrachialPlex_R',
    'Brain', 'Brainstem', 'Bronchus', 'Breast_L',
    'Breast_R', 'Chestwall', 'Cochlea_L', 'Cochlea_R', 'Colon_Sigmoid', 'Duodenum', 'Esophagus', 'Femur_Head_L',
    'Femur_Head_R', 'Gallbladder', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Heart', 'Kidney_L', 'Kidney_R',
    'Larynx', 'Liver', 'Lung_L', 'Lung_R', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 
    'Pericardium', 'Rectum', 'Skin',
    'SpinalCanal', 'SpinalCord', 'Stomach', 'Trachea'
]
anonymise = False

convert_to_nifti(dataset, region=regions, anonymise=anonymise)
