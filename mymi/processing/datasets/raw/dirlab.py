import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from mymi import config
from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate
from mymi.utils import *

def convert_lung_4dct_to_nifti() -> None:
    dataset = 'DIRLAB-LUNG-4DCT'
    phase_1 = 'T00'
    phase_2 = 'T50'
    dry_run = False
    shift_data = True
    shift = -1024
    truncate_data = True
    data_min, data_max = (-1024, 1000)

    # Convert to our NiftiDataset format.
    rset = RawDataset(dataset)
    filepath = os.path.join(rset.path, 'metadata.csv')
    df = pd.read_csv(filepath)

    set = recreate(dataset)
    filepath = os.path.join(set.path, 'data')
    files = list(sorted(os.listdir(filepath)))
    for f in tqdm(files):
        # print(pat_id)
        pat_id = f.lower().replace('pack', '')
        pat_info = df[df['patient-id'] == pat_id].iloc[0]
        size = tuple(pat_info[['size-x', 'size-y', 'size-z']].tolist())
        spacing = tuple(pat_info[['spacing-x', 'spacing-y', 'spacing-z']].tolist())
        
        # Inhale.
        filepath = os.path.join(set.path, 'data', f, 'Images', f'{pat_id}_{phase_1}-ssm.img')
        img = __read_lung_4dct_image(filepath, size, sitk.sitkInt16, image_spacing=spacing)
        edata = sitk.GetArrayFromImage(img)
        edata = np.moveaxis(np.moveaxis(edata, 0, -1), 0, 1)
        edata = np.flip(edata, 2)
        if shift_data:
            edata += shift
        if truncate_data:
            edata[edata > data_max] = data_max
            edata[edata < data_min] = data_min
        offset = (0, 0, 0)
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, 'study_0', 'ct', 'series_0.nii.gz')
            save_as_nifti(edata, spacing, offset, filepath)
            
        # Inhale points.
        filepath = os.path.join(set.path, 'data', f, 'ExtremePhases', f'{pat_id.capitalize()}_300_{phase_1}_xyz.txt')
        epoints = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        epoints = epoints[[0, 1, 2]]
        epoints[2] = size[2] - epoints[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        epointsdata = epoints[list(range(3))].to_numpy()
        epointsdata = epointsdata * spacing
        epoints[list(range(3))] = epointsdata
        epoints.insert(0, 'landmark-id', list(range(len(epoints))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, 'study_0', 'landmarks', 'series_1.csv')
            save_csv(epoints, filepath, header=True, index=False)
        
        # Exhale.
        filepath = os.path.join(set.path, 'data', f, 'Images', f'{pat_id}_{phase_2}-ssm.img')
        img = __read_lung_4dct_image(filepath, size, sitk.sitkInt16, image_spacing=spacing)
        idata = sitk.GetArrayFromImage(img)
        idata = np.moveaxis(np.moveaxis(idata, 0, -1), 0, 1)
        idata = np.flip(idata, 2)
        if shift_data:
            idata += shift
        if truncate_data:
            idata[idata > data_max] = data_max
            idata[idata < data_min] = data_min
        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, 'study_1', 'ct', 'series_0.nii.gz')
            offset = (0, 0, 0)
            save_as_nifti(idata, spacing, offset, filepath)
            
        # Exhale points.
        filepath = os.path.join(set.path, 'data', f, 'ExtremePhases', f'{pat_id.capitalize()}_300_{phase_2}_xyz.txt')
        ipoints = pd.read_csv(filepath, sep='\t', header=None, engine='python')
        ipoints = ipoints[[0, 1, 2]]
        ipoints[2] = size[2] - ipoints[2]
        
        # Convert landmarks from image coordinates to patient coordinates.
        ipointsdata = ipoints[list(range(3))].to_numpy()
        ipointsdata = ipointsdata * spacing
        ipoints[list(range(3))] = ipointsdata
        ipoints.insert(0, 'landmark-id', list(range(len(ipoints))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', pat_id, 'study_1', 'landmarks', 'series_1.csv')
            save_csv(ipoints, filepath, header=True, index=False)

def __read_lung_4dct_image(
    binary_file_name,
    image_size,
    sitk_pixel_type,
    image_spacing=None,
    image_origin=None,
    big_endian=False,
):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
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
                " ".join([str(v) for v in image_spacing])
                if image_spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in image_origin])
                if image_origin
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
    
    # print(header)
    
    filepath = os.path.join(config.directories.temp, f"tmp-{binary_file_name.split('/')[-1].replace('.img', '')}.mhd")
    with open(filepath, 'wb') as tmp:
        tmp.writelines(header)
        tmp.close()
        img = sitk.ReadImage(tmp.name)
    return img
