from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)

from mymi.processing.dataset.dicom import upload_rtstructs_from_all_patients

N_PATS = 50
INTERVAL = 30

upload_rtstructs_from_all_patients(N_PATS, interval=INTERVAL)
