import fire
import itk
import nibabel as nib
import numpy as np
import os
import subprocess
import sys
import torch
from tqdm import tqdm
from typing import Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi import config
from mymi.dataset.nifti import NiftiDataset
from mymi import logging
from mymi.regions import region_to_list
from mymi.types import PatientRegions

def predict(
    dataset: str,
    model: str,
    modelname: str,
    register_images: bool = True,
    region: Optional[PatientRegions] = None) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'unigradicon-index-paired.csv')
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = [l.strip() for l in lines]
    lines = [l.split(' ') for l in lines]

    # Create moved image.
    if register_images:
        logging.info('Making predictions...')
        os.makedirs(os.path.join(set.path, 'predictions', modelname, 'ct'), exist_ok=True)
        for movingpath, fixedpath in tqdm(lines):
            movedpath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath))
            warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.hdf5'))
            # Call voxelmorph script.
            subprocess.run([
                'unigradicon-register',
                '--moving', movingpath,
                '--moving_modality', 'ct',
                '--fixed', fixedpath,
                '--fixed_modality', 'ct',
                '--warped_moving_out', movedpath,
                '--transform_out', warppath
            ]) 

    # Apply warp to any segmentation labels.
    regions = region_to_list(region)
    if regions is not None:
        logging.info('Making label predictions...')
        for region in tqdm(regions):
            os.makedirs(os.path.join(set.path, 'predictions', modelname, 'regions', region), exist_ok=True)
            for movingpath, fixedpath in tqdm(lines, leave=False):
                movinglabelpath = movingpath.replace('/ct/', f'/regions/{region}/')
                movedlabelpath = os.path.join(set.path, 'predictions', modelname, 'regions', region, os.path.basename(movingpath))
                warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.hdf5'))
                # Call voxelmorph script.
                subprocess.run([
                    'unigradicon-warp',
                    '--moving', movinglabelpath,
                    '--fixed', fixedpath,
                    '--transform', warppath,
                    '--warped_moving_out', movedlabelpath,
                    '--nearest_neighbor'
                ])  

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
        warppath = os.path.join(set.path, 'predictions', modelname, 'ct', os.path.basename(movingpath).replace('.nii.gz', '_warp.hdf5'))
        warp = itk.transformread(warppath)
        assert len(warp) == 1
        warp = warp[0]

        # Get coordinate transforms for warping.
        fixedpath = fixed_pat.ct_path
        fixed_itk = itk.imread(fixedpath)
        origin = np.array(fixed_itk.GetOrigin())
        spacing = np.array(fixed_itk.GetSpacing())
        direction = np.array(fixed_itk.GetDirection())
        movingpath = moving_pat.ct_path
        moving_itk = itk.imread(movingpath)
        origin_m = np.array(moving_itk.GetOrigin())
        spacing_m = np.array(moving_itk.GetSpacing())
        direction_m = np.array(moving_itk.GetDirection())
        # Deal with moving/fixed differences if it becomes a problem...
        assert np.array_equal(origin_m, origin)
        assert np.array_equal(spacing_m, spacing)
        assert np.array_equal(direction_m, direction)

        # Transform points.
        points_t = []
        for point in fixed_lms:
            # Transform points to physical coordinates for ITK transform
            point_mm = point_mm = origin + direction.dot(point * spacing)

            # Perform warp.
            point_t = warp.TransformPoint(point_mm)

            # Transform to moving space voxel coords - assuming same coordinate transform
            # for moving and fixed images.
            inv_direction = np.linalg.inv(direction)
            point_t = inv_direction.dot(point_t - origin) / spacing
            points_t.append(point_t)
        points_t = np.vstack(points_t)

        # Save transformed points.
        filepath = os.path.join(set.path, 'predictions', modelname, 'landmarks', f'{moving_pat_id}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, points_t, delimiter=',', fmt='%.3f')
 
fire.Fire(predict)
