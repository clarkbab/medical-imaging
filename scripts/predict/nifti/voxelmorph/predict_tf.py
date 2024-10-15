import fire
import nibabel as nib
import numpy as np
import os
import subprocess
import sys
import tensorflow as tf
from tqdm import tqdm
from typing import Optional

VXMPATH="/home/baclark/code/voxelmorph"
sys.path.append(VXMPATH)
from voxelmorph.tf.layers import SpatialTransformer

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
    modelname: str,
    register_images: bool = True,
    region: Optional[PatientRegions] = None) -> None:
    modelpath = os.path.join(config.directories.models, 'voxelmorph', model)
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'voxelmorph-index-paired.csv')
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split(' ') for l in lines]

    if register_images:
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'ct'), exist_ok=True)
        logging.info('Making predictions...')
        for movingpath, fixedpath in tqdm(lines):
            movedpath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath))
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
            # Call voxelmorph script.
            command = [
                'python', os.path.join(VMXPATH, 'scripts', 'tf', 'register_semisupervised_seg.py'),
                '--moving', movingpath,
                '--fixed', fixedpath,
                '--moved', movedpath,
                '--warp', warppath,
                '--model', modelpath,  
                '--gpu', '0'
            ] 
            logging.info(command)
            subprocess.run(command)

    # Apply warp to any segmentation labels.
    regions = region_to_list(region)
    if regions is not None:
        logging.info('Warping labels...')
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'regions'), exist_ok=True)

        # Create warper layer.
        labelpath = os.path.join(set.path, 'data', 'regions', regions[0], os.path.basename(lines[0][0]))
        label = nib.load(labelpath).get_fdata()
        warper = SpatialTransformer(interp_method='linear')

        for movingpath, fixedpath in tqdm(lines):
            # Load the warp. Make it so.
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
            warp = nib.load(warppath).get_fdata()

            # Load labels, apply warp and save. 
            for region in regions:
                labelpath = os.path.join(set.path, 'data', 'regions', region, os.path.basename(movingpath))
                logging.info(labelpath)
                label = nib.load(labelpath).get_fdata()
                # Label requires channel dimension last - tensorflow default.
                label = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(label), 0), -1)
                warp = tf.expand_dims(tf.convert_to_tensor(warp), 0)
                warped = warper([label, warp])
                logging.info(warped.shape)
                warped = tf.squeeze(tf.squeeze(warped, 0), -1).numpy()
                logging.info(warped.shape)
                warpedpath = os.path.join(set.path, 'predictions', modelname, 'regions', region, os.path.basename(movingpath))
                os.makedirs(os.path.dirname(warpedpath), exist_ok=True)
                nib.save(nib.Nifti1Image(warped, np.eye(4)), warpedpath)
 
fire.Fire(predict)
