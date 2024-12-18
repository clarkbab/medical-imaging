from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.nifti import create_registered_dataset

dataset = 'PMCC-REIRRAD-NOALIGN'
dest_dataset = 'PMCC-REIRRAD'
landmarks = 'all'
regions = f'RL:PMCC-REIRRAD'

create_registered_dataset(dataset, dest_dataset, landmarks=landmarks, regions=regions)
