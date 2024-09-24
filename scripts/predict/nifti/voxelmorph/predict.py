import fire
import nibabel as nib
import numpy as np
import os
import subprocess
import sys
import torch
from tqdm import tqdm
from typing import Optional

VXMPATH="/home/baclark/code/voxelmorph"
sys.path.append(VXMPATH)
from voxelmorph.torch.layers import SpatialTransformer

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi import config
from mymi.dataset.nifti import NiftiDataset
from mymi import logging
from mymi.regions import region_to_list
from mymi.types import PatientRegions

VMXPATH="/home/baclark/code/voxelmorph"

def predict(
    dataset: str,
    model: str,
    labels_only: bool = False,
    region: Optional[PatientRegions] = None) -> None:
    modelname = model.split('/')[0]
    modelpath = os.path.join(config.directories.models, 'voxelmorph', model)
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'voxelmorph-index-paired.csv')
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split(' ') for l in lines]

    if not labels_only:
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'ct'), exist_ok=True)
        logging.info('Making predictions...')
        for movingpath, fixedpath in tqdm(lines):
            movedpath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath))
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
            # Call voxelmorph script.
            subprocess.run([
                'python', os.path.join(VMXPATH, 'scripts', 'torch', 'register.py'),
                '--moving', movingpath,
                '--fixed', fixedpath,
                '--moved', movedpath,
                '--warp', warppath,
                '--model', modelpath,  
                '--gpu', '0'
            ]) 

    # Apply warp to any segmentation labels.
    regions = region_to_list(region)
    if regions is not None:
        logging.info('Warping labels...')
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'regions'), exist_ok=True)

        # Create warper layer.
        labelpath = os.path.join(set.path, 'data', 'regions', regions[0], os.path.basename(lines[0][0]))
        label = nib.load(labelpath).get_fdata()
        warper = SpatialTransformer(label.shape, mode='bilinear')

        for movingpath, fixedpath in tqdm(lines):
            # Load the warp. Make it so.
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
            warp = nib.load(warppath).get_fdata()

            # Load labels, apply warp and save. 
            for region in regions:
                labelpath = os.path.join(set.path, 'data', 'regions', region, os.path.basename(movingpath))
                label = nib.load(labelpath).get_fdata()
                warped = warper(torch.Tensor(label).unsqueeze(0).unsqueeze(0), torch.Tensor(warp).unsqueeze(0))
                warped = warped.squeeze().squeeze().detach().numpy()
                warpedpath = os.path.join(set.path, 'predictions', modelname, 'regions', region, os.path.basename(movingpath))
                os.makedirs(os.path.dirname(warpedpath), exist_ok=True)
                nib.save(nib.Nifti1Image(warped, np.eye(4)), warpedpath)
 
fire.Fire(predict)
