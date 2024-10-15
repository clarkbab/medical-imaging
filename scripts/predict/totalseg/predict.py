import fire
import os
import subprocess
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import logging

def predict(dataset: str) -> None:
    basepath = os.path.join('/data/gpfs/projects/punim1413/mymi/totalseg/datasets', dataset, 'data')
    ctpath = os.path.join(basepath, 'ct')
    os.makedirs(ctpath, exist_ok=True)
    for file in sorted(os.listdir(ctpath)):
        srcpath = os.path.join(ctpath, file)
        destpath = os.path.join(basepath, 'regions', file.replace('.nii.gz', ''))
        command = [
            'TotalSegmentator',
            '-i', srcpath,
            '-o', destpath
        ]
        logging.info(command)
        subprocess.run(command) 
 
fire.Fire(predict)
