import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label as label_objects

from mymi import dataset as ds

from .extent import get_extent, get_extent_centre

def get_object_summary(
    dataset: str,
    patient: str,
    region: str) -> pd.DataFrame:
    pat = ds.get(dataset, 'nifti').patient(patient)
    spacing = pat.ct_spacing()
    label = pat.region_data(regions=region)[region]
    objs, num_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = _one_hot_encode(objs)
    
    cols = {
        'extent-centre-vox': str,
        'extent-width-vox': str,
        'volume-mm3': float,
        'volume-p': float,
        'volume-vox': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    tot_voxels = label.sum()
    for i in range(num_objs):
        obj = objs[:, :, :, i]
        data = {}

        # Get extent.
        min, max = get_extent(obj)
        width = tuple(np.array(max) - min)
        data['extent-width-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = get_extent_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        num_voxels = obj.sum()
        volume = num_voxels * vox_volume
        data['volume-vox'] = num_voxels
        data['volume-p'] = num_voxels / tot_voxels
        data['volume-mm3'] = volume

        df = df.append(data, ignore_index=True)

    df = df.astype(cols)
    return df

def get_object(
    dataset: str,
    patient: str,
    region: str,
    id: int) -> np.ndarray:
    pat = ds.get(dataset, 'nifti').patient(patient)
    label = pat.region_data(regions=region)[region]
    objs, num_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = _one_hot_encode(objs)
    return objs[:, :, :, id]

def _one_hot_encode(a):
    return (np.arange(a.max()) == a[...,None]-1).astype(bool)
