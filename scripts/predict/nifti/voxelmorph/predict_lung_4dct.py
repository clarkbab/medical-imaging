import fire
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import subprocess
import sys
import torch
from tqdm import tqdm
from typing import Optional

VXMPATH="/home/baclark/code/voxelmorph"
sys.path.append(VXMPATH)
# from voxelmorph.torch.layers import SpatialTransformer

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi import config
from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import centre_crop_or_pad
from mymi.typing import Regions

VMXPATH="/home/baclark/code/voxelmorph"

def predict(
    dataset: str,
    model: str,
    modelname: str,
    crop_images: bool = False,
    register_images: bool = True,
    region: Optional[Regions] = None) -> None:
    modelpath = os.path.join(config.directories.models, 'voxelmorph', model)
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'voxelmorph-index-paired.csv')
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split(' ') for l in lines]

    if register_images:
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'ct'), exist_ok=True)
        logging.info('Making predictions (normalised)...')
        for movingpath, fixedpath in tqdm(lines):
            movedpath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath))
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
            os.makedirs(os.path.dirname(movedpath), exist_ok=True)

            # Create temporary images that have been preprocessed for Voxelmorph:
            # - Cropped/padded to [192, 192, 208] to match L2R-LUNG training dataset.
            # - Normalised to [0, 1].
            tmpmovingpath = movingpath.replace('.nii.gz', '_vxm.nii.gz')
            img = nib.load(movingpath)
            data = img.get_fdata()
            original_shape = data.shape
            print(f"original shape = {original_shape}")
            shape = (192, 192, 208)
            if crop_images:
                data = centre_crop_or_pad(data, shape)
            print(f"cropped shape = {data.shape}")
            moving_min, moving_max = np.min(data), np.max(data)
            data = (data - moving_min) / (moving_max - moving_min)
            img = Nifti1Image(data, img.affine)
            nib.save(img, tmpmovingpath)

            tmpfixedpath = fixedpath.replace('.nii.gz', '_vxm.nii.gz')
            img = nib.load(fixedpath)
            data = img.get_fdata()
            if crop_images:
                data = centre_crop_or_pad(data, shape)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            img = Nifti1Image(data, img.affine)
            nib.save(img, tmpfixedpath)

            # Call voxelmorph script.
            subprocess.run([
                'python', os.path.join(VMXPATH, 'scripts', 'torch', 'register.py'),
                '--moving', tmpmovingpath,
                '--fixed', tmpfixedpath,
                '--moved', movedpath,
                '--warp', warppath,
                '--model', modelpath,  
                '--gpu', '0'
            ]) 

            # Unnormalise the moved image.
            img = nib.load(movedpath)
            data = img.get_fdata()
            print(f"predicted shape = {data.shape}")
            data = data * (moving_max - moving_min) + moving_min
            if crop_images:
                data = centre_crop_or_pad(data, original_shape)
            print(f"padded shape = {data.shape}")
            img = Nifti1Image(data, img.affine)
            nib.save(img, movedpath)

    # Apply warp to any segmentation labels.
    regions = regions_to_list(region)
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

    # Transform any fixed landmarks back to moving space.
    logging.info('Transforming landmarks...')
    for movingpath, fixedpath in tqdm(lines):
        fixed_pat_id = os.path.basename(fixedpath).split('.')[0]
        fixed_pat = set.patient(fixed_pat_id)
        moving_pat_id = os.path.basename(movingpath).split('.')[0]
        moving_pat = set.patient(moving_pat_id)
        if not fixed_pat.has_landmarks:
            logging.info(f'No landmarks found for patient {fixed_pat_id}. Skipping...')
            continue
        fixed_lms = fixed_pat.landmarks
        moving_pat_id = os.path.basename(movingpath).split('.')[0]

        # Load transform.
        # Warp has same shape as the fixed image and also is in voxel coordinates, not normalised
        # voxel coordinates as required by Torch's "grid_resample".
        warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.nii.gz'))
        warp = nib.load(warppath).get_fdata()

        # Transform points.
        points_t = []
        for point in fixed_lms:
            idx = tuple([slice(None)] + list(point))
            point_t = point + warp[idx]
            points_t.append(point_t)
        points_t = np.vstack(points_t)

        # Save transformed points.
        filepath = os.path.join(set.path, 'predictions', modelname, 'landmarks', f'{moving_pat_id}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, points_t, delimiter=',', fmt='%.3f')
 
fire.Fire(predict)
