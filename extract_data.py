import numpy as np
import pydicom as dicom
from tqdm import tqdm
import os
from utils import load_ct_dicoms, load_rtstruct_dicom
from skimage.draw import polygon
import argparse

# Parse args.
parser = argparse.ArgumentParser(description='Convert DICOM to input-label data')
parser.add_argument('-o', '--overwrite', type=bool, required=False, default=False, help='overwrite existing converted data')
args = parser.parse_args()

DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1')
DATA_SOURCE = os.path.join(DATA_ROOT, 'raw')
DATA_DEST = os.path.join(DATA_ROOT, 'processed')

def already_converted(pat_path):
    """
    Checks if data files already exist.
    pat_path: path to the processed data folder for patient.
    """
    return os.path.isfile(os.path.join(pat_path, 'input.npy'))

def get_ct_info(dicoms):
    """
    Returns info about CT scans such as pixel spacing and offset.
    dicoms: the list of CT dicoms.
    """
    info = {}
    dicom = dicoms[0]

    # Load position info.
    info['pos_x'] = dicom.ImagePositionPatient[0]
    info['pos_y'] = dicom.ImagePositionPatient[1]
    pos_zs = [round(d.ImagePositionPatient[2], 1) for d in dicoms] # Must round as CT z-axis precision has more d.p. than RTSTRUCT coordinates.
    pos_zs.sort()
    info['pos_zs'] = pos_zs

    # Load spacing info.
    info['spacing_x'] = dicom.PixelSpacing[0]
    info['spacing_y'] = dicom.PixelSpacing[1]

    # Load dimensions info.
    info['ct_dimensions'] = tuple(np.append(np.array(dicom.pixel_array.shape), len(pos_zs)))

    return info

def convert_to_pixel(dicoms):
    """
    Converts RTSTRUCT DICOM files to pixel data.
    dicoms: a list of RTSTRUCT DICOM files.
    """
    # Pull pixel data.
    pixel_data = np.stack([d.pixel_array for d in dicoms], axis=-1)

    # Convert to pixel space to align with CT data.
    intercept = dicoms[0].RescaleIntercept
    slope = dicoms[0].RescaleSlope
    pixel_data = slope * pixel_data + intercept
    pixel_data = pixel_data.astype(np.int16)

    return pixel_data

def convert_to_labels(struct_dicom, ct_info):
    """
    Converts rtstruct dicom file to label data.
    struct_dicom: the rtstruct dicom file.
    pos_info: positioning info for the CT scans.
    """
    # Load all ROIs.
    roi_infos = struct_dicom.StructureSetROISequence
    rois = struct_dicom.ROIContourSequence

    # Will store all labels, assuming that no ROI overlap can occur.
    labels = []
    colors = {}

    # Test with first ROI.
    for roi, roi_info in zip(rois, roi_infos):
        # Create placeholder label.
        label = np.zeros(shape=ct_info['ct_dimensions'], dtype=np.uint8)

        roi_num = int(roi_info.ROINumber)
        roi_coords = [c.ContourData for c in roi.ContourSequence]
        colors[roi_num] = roi.ROIDisplayColor

        for roi_slice_coords in roi_coords:
            # Coords are stored in flat array.
            coords = np.array(roi_slice_coords).reshape(-1, 3)

            # Check that z-axis values are the same for the slice.
            assert len(np.unique(coords[:, 2])) == 1
            pos_z = coords[0, 2]

            # Get slice index, relative to CT data.
            pixel_z = ct_info['pos_zs'].index(pos_z)

            # Convert from "real" space to pixel space using affine transformation.
            pixel_x = (coords[:, 0] - ct_info['pos_x']) / ct_info['spacing_x']
            pixel_y = (coords[:, 1] - ct_info['pos_y']) / ct_info['spacing_y']

            # Get 2D coords of polygon boundary and interior described by corner
            # points.
            xx, yy = polygon(pixel_x, pixel_y)

            # Set labelled pixels in slice.
            label[yy, xx, pixel_z] = 1


        labels.append((roi_num, label))

    return labels

def save_data(pat_path, input, labels):
    """
    Saves the CT/RTSTRUCT data for a patient.
    pat_path: the path to the patient data.
    input: the input data (CT scan pixel data).
    labels: the label data (a 3D tensor for each ROI).
    """
    labels_path = os.path.join(pat_path, 'labels')
    os.makedirs(labels_path, exist_ok=True)

    # Save inputs.
    input_path = os.path.join(pat_path, 'input.npy') 
    np.save(input_path, input) 

    # Save labels.
    for roi_num, label in labels:
        label_path = os.path.join(labels_path, f"label_{roi_num}.npy")
        np.save(label_path, label)
        

# List patients.
pat_dirs = os.listdir(DATA_SOURCE)
pat_dirs.sort()
 
# For each patient.
for pat_dir in tqdm(pat_dirs):
    # Get path to patient folder.
    pat_path = os.path.join(DATA_SOURCE, pat_dir)
    print(f"Copying for patient: {pat_path}")

    # Skip patient if alread processed.
    pat_proc_path = os.path.join(DATA_DEST, pat_dir)
    if not args.overwrite and already_converted(pat_proc_path):
        print("Already converted, skipping.")
        continue

    # Load CT dicoms.
    ct_dicoms = load_ct_dicoms(pat_path)

    # Convert to pixel data.
    input_data = convert_to_pixel(ct_dicoms)

    # Load RTSTRUCT dicom.
    rtstruct_dicom = load_rtstruct_dicom(pat_path)
    
    # Get scan settings.
    ct_info = get_ct_info(ct_dicoms)

    # Convert to label data.
    labels_data = convert_to_labels(rtstruct_dicom, ct_info)

    # Save raw data.
    save_data(pat_proc_path, input_data, labels_data)

exit(0)

