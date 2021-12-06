from os.path import dirname as up
import numpy as np
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi import dataset as ds
from mymi.metrics import dice, distances

# Get patients.
dataset1 = 'PMCC-HN-TRAIN'
dataset2 = 'PMCC-HN-TEST'
regions = ('BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R')
pats1 = ds.get(dataset1, 'dicom').list_patients(regions=regions)
pats2 = ds.get(dataset2, 'dicom').list_patients(regions=regions)

# Get intersection.
intersect = np.intersect1d(pats1, pats2)
if len(intersect) != 0:
    raise ValueError(f"Overlapping patients '{list(intersect)}'.")
