import fire
import os
import subprocess
import sys
from tqdm import tqdm
from typing import List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import logging
from mymi.utils import arg_to_list

def predict(
    dataset: str,
    tasks: List[str] = 'total') -> None:
    tasks = arg_to_list(tasks, str)
    basepath = os.path.join('/data/gpfs/projects/punim1413/mymi/totalseg/datasets', dataset, 'data')
    ctpath = os.path.join(basepath, 'ct')
    os.makedirs(ctpath, exist_ok=True)
    for file in sorted(os.listdir(ctpath)):
        srcpath = os.path.join(ctpath, file)
        for task in tasks:
            destpath = os.path.join(basepath, 'regions', file.replace('.nii.gz', ''))
            command = [
                'TotalSegmentator',
                '-i', srcpath,
                '-o', destpath,
                '--task', task
            ]
            logging.info(command)
            subprocess.run(command) 
 
fire.Fire(predict)
