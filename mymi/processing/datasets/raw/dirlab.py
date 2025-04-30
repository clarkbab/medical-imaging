import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from mymi import config
from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate
from mymi.utils import *

from ...processing import fill_border_padding, fill_contiguous_padding

def convert_lung_copd_to_nifti() -> None:
    dataset = 'DIRLAB-LUNG-COPD'
    dry_run = False
    shift_data = True
    shift = -1024
    truncate_data = True
    data_min, data_max = (-1024, 2000)
    replace_padding = True
    fill = -2000

    # Convert to our NiftiDataset format.
    rset = RawDataset(dataset)
    datapath = os.path.join(rset.path, 'data')
    filepath = os.path.join(datapath, 'copd_metadata.csv')
    df = pd.read_csv(filepath)

    set = recreate(dataset)
    pat_ids = os.listdir(datapath)
    pat_ids = [p for p in pat_ids if p.startswith('copd') and p != 'copd_metadata.csv']
    pat_ids = list(sorted(pat_ids))
    for p in tqdm(pat_ids):
        pat_info = df[df['Label'] == p].iloc[0]
        size = tuple(pat_info[['image_dims0', 'image_dims1', 'image_dims2']].tolist())
        spacing = tuple(pat_info[['vspacing0', 'vspacing1', 'vspacing2']].tolist())
        
        # Inhale - should be fixed image.
        fixed_study = 'study_1'
        filepath = os.path.join(datapath, p, f'{p}_iBHCT.img')
        data = read_dirlab_image(filepath, size, spacing, sitk.sitkInt16)
        if shift_data:
            data += shift
        if replace_padding:
            data = fill_border_padding(data, fill=fill)
            data = fill_contiguous_padding(data, fill=fill)
        # Truncate after padding to avoid creating connected regions.
        if truncate_data:
            data[(data != fill) & (data > data_max)] = data_max
            data[(data != fill) & (data < data_min)] = data_min
        offset = (0, 0, 0)
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'ct', 'series_0.nii.gz')
            save_as_nifti(data, spacing, offset, filepath)
            
        # Inhale points.
        filepath = os.path.join(rset.path, 'data', p, f'{p}_300_iBH_xyz_r1.txt')
        points = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        points = points[[0, 1, 2]]
        points[2] = size[2] - points[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        points_data = points[list(range(3))].to_numpy()
        points_data = spacing * points_data
        points[list(range(3))] = points_data
        points.insert(0, 'landmark-id', list(range(len(points))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'landmarks', 'series_1.csv')
            save_files_csv(points, filepath, header=True, index=False)
        
        # Exhale - should be moving image.
        moving_study = 'study_0'
        filepath = os.path.join(datapath, p, f'{p}_eBHCT.img')
        data = read_dirlab_image(filepath, size, spacing, sitk.sitkInt16)
        if shift_data:
            data += shift
        if replace_padding:
            data = fill_border_padding(data, fill=fill)
            data = fill_contiguous_padding(data, fill=fill)
        # Truncate after padding to avoid creating connected regions.
        if truncate_data:
            data[(data != fill) & (data > data_max)] = data_max
            data[(data != fill) & (data < data_min)] = data_min
        offset = (0, 0, 0)
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'ct', 'series_0.nii.gz')
            save_as_nifti(data, spacing, offset, filepath)
            
        # Exhale points.
        filepath = os.path.join(rset.path, 'data', p, f'{p}_300_eBH_xyz_r1.txt')
        points = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        points = points[[0, 1, 2]]
        points[2] = size[2] - points[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        points_data = points[list(range(3))].to_numpy()
        points_data = spacing * points_data
        points[list(range(3))] = points_data
        points.insert(0, 'landmark-id', list(range(len(points))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'landmarks', 'series_1.csv')
            save_files_csv(points, filepath, header=True, index=False)

def convert_lung_4dct_to_nifti() -> None:
    dataset = 'DIRLAB-LUNG-4DCT'
    phase_1 = 'T00'
    phase_2 = 'T50'
    dry_run = False
    shift_data = True
    shift = -1024
    truncate_data = True
    data_min, data_max = (-1024, 2000)
    replace_padding = True
    fill = -2000

    # Convert to our NiftiDataset format.
    rset = RawDataset(dataset)
    filepath = os.path.join(rset.path, 'metadata.csv')
    df = pd.read_csv(filepath)

    set = recreate(dataset)
    filepath = os.path.join(rset.path, 'data')
    files = list(sorted(os.listdir(filepath)))
    for f in tqdm(files):
        # print(pat_id)
        pat_id = f.lower().replace('pack', '')
        pat_info = df[df['patient-id'] == pat_id].iloc[0]
        size = tuple(pat_info[['size-x', 'size-y', 'size-z']].tolist())
        spacing = tuple(pat_info[['spacing-x', 'spacing-y', 'spacing-z']].tolist())
        
        # Inhale - should be fixed image.
        inhale_study = 'study_1'
        filepath = os.path.join(rset.path, 'data', f, 'Images', f'{pat_id}_{phase_1}-ssm.img')
        data = read_dirlab_image(filepath, size, spacing, sitk.sitkInt16)
        if shift_data:
            data += shift
        if replace_padding:
            data = fill_border_padding(data, fill=fill)
            data = fill_contiguous_padding(data, fill=fill)
        # Truncate after padding to avoid creating connected regions.
        if truncate_data:
            data[(data != fill) & (data > data_max)] = data_max
            data[(data != fill) & (data < data_min)] = data_min
        offset = (0, 0, 0)
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, 'study_1', 'ct', 'series_0.nii.gz')
            save_as_nifti(data, spacing, offset, filepath)
            
        # Inhale points.
        filepath = os.path.join(rset.path, 'data', f, 'ExtremePhases', f'{pat_id.capitalize()}_300_{phase_1}_xyz.txt')
        points = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        points = points[[0, 1, 2]]
        points[2] = size[2] - points[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        points_data = points[list(range(3))].to_numpy()
        points_data = points_data * spacing
        points[list(range(3))] = points_data
        points.insert(0, 'landmark-id', list(range(len(points))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, inhale_study, 'landmarks', 'series_1.csv')
            save_files_csv(points, filepath, header=True, index=False)
        
        # Exhale - should be moving image.
        exhale_study = 'study_0'
        filepath = os.path.join(rset.path, 'data', f, 'Images', f'{pat_id}_{phase_2}-ssm.img')
        data = read_dirlab_image(filepath, size, spacing, sitk.sitkInt16)
        if shift_data:
            data += shift
        if replace_padding:
            data = fill_border_padding(data, fill=fill)
            data = fill_contiguous_padding(data, fill=fill)
        # Truncate after padding to avoid creating connected regions.
        if truncate_data:
            data[(data != fill) & (data > data_max)] = data_max
            data[(data != fill) & (data < data_min)] = data_min
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, exhale_study, 'ct', 'series_0.nii.gz')
            offset = (0, 0, 0)
            save_as_nifti(data, spacing, offset, filepath)
            
        # Exhale points.
        filepath = os.path.join(rset.path, 'data', f, 'ExtremePhases', f'{pat_id.capitalize()}_300_{phase_2}_xyz.txt')
        points = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        points = points[[0, 1, 2]]
        points[2] = size[2] - points[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        points_data = points[list(range(3))].to_numpy()
        points_data = points_data * spacing
        points[list(range(3))] = points_data
        points.insert(0, 'landmark-id', list(range(len(points))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, exhale_study, 'landmarks', 'series_1.csv')
            save_files_csv(points, filepath, header=True, index=False)

def read_dirlab_image(
    binary_file_name,
    image_size,
    spacing,
    sitk_pixel_type,
    offset: Optional[Voxel] = None,
    big_endian=False) -> np.ndarray:
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    offset (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 -1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]
    dim = len(image_size)
    header = [
        "ObjectType = Image\n".encode(),
        (f"NDims = {dim}\n").encode(),
        ("DimSize = " + " ".join([str(v) for v in image_size]) + "\n").encode(),
        (
            "ElementSpacing = "
            + (
                " ".join([str(v) for v in spacing])
                if spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in offset])
                if offset
                else " ".join(["0"] * dim) + "\n"
            )
        ).encode(),
        ("TransformMatrix = " + direction_cosine[dim - 2] + "\n").encode(),
        ("ElementType = " + pixel_dict[sitk_pixel_type] + "\n").encode(),
        "BinaryData = True\n".encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        # ElementDataFile must be the last entry in the header
        ("ElementDataFile = " + os.path.abspath(binary_file_name) + "\n").encode(),
    ]
    
    filepath = os.path.join(config.directories.temp, f"tmp-{binary_file_name.split('/')[-1].replace('.img', '')}.mhd")
    with open(filepath, 'wb') as tmp:
        tmp.writelines(header)
        tmp.close()
        img = sitk.ReadImage(tmp.name)

    # Convert from sitk to our format.
    data, _, _ = from_sitk_image(img)

    return data
