import argparse
import numpy as np
import os
import pydicom as dicom
from skimage.draw import polygon
from tqdm import tqdm
import utils

# Parse args.
parser = argparse.ArgumentParser(description='Convert DICOM to input-label data')
parser.add_argument('-o', '--overwrite', required=False, default=False, action='store_true', help='overwrite existing converted data')
parser.add_argument('-n', '--numpats', type=int, required=False, help='run on a smaller number of patients' )
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

def get_labels(rtstruct_dicom, ct_dicoms_summary_info_df):
    """
    Converts RTSTRUCT dicom file to labelled data.
    rtstruct_dicom: the RTSTRUCT dicom object.
    ct_dicoms_summary_info_df: a dataframe containing summary data of CT scans.
        dim-x: the x dimensions of the CT scan volume.
        dim-y: the y dimensions of the CT scan volume.
        dim-z: the z dimensions of the CT scan volume.
        offset-x: the x offset of each CT scan.
        offset-y: the y offset of each CT scan.
        res-x: the x-direction pixel resolution of the CT scan volume.
        res-y: the y-direction pixel resolution of the CT scan volume.
    """
    # Load all ROIs.
    roi_infos = rtstruct_dicom.StructureSetROISequence
    rois = rtstruct_dicom.ROIContourSequence

    # Will store all labels, assuming that no ROI overlap can occur.
    labels = []

    # Create a label for each ROI.
    for roi, roi_info in zip(rois, roi_infos):
        # Create label placeholder.
        label_shape = (ct_dicoms_info_summary_df['dim-x'][0], ct_dicoms_info_summary_df['dim-y'][0], ct_dicoms_info_summary_df['dim-z'][0])
        label = np.zeros(shape=label_shape, dtype=np.uint8)

        roi_name = roi_info.ROIName
        roi_coords = [c.ContourData for c in roi.ContourSequence]

        # Label each slice of the ROI.
        for roi_slice_coords in roi_coords:
            # Coords are stored in flat array.
            coords = np.array(roi_slice_coords).reshape(-1, 3)

            # Convert from "real" space to pixel space using affine transformation.
            corner_pixels_x = (coords[:, 0] - ct_dicoms_info_summary_df['offset-x'][0]) / ct_dicoms_info_summary_df['res-x'][0]
            corner_pixels_y = (coords[:, 1] - ct_dicoms_info_summary_df['offset-y'][0]) / ct_dicoms_info_summary_df['res-y'][0]

            # Get contour z pixel.
            offset_z = coords[0, 2] - ct_dicoms_info_summary_df['offset-z'][0]
            pixel_z = int(offset_z / ct_dicoms_info_summary_df['res-z'][0])

            # Get 2D coords of polygon boundary and interior described by corner
            # points.
            pixels_x, pixels_y = polygon(corner_pixels_x, corner_pixels_y)

            # Set labelled pixels in slice.
            label[pixels_x, pixels_y, pixel_z] = 1

        labels.append((roi_name, label))

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
    for roi_name, label in labels:
        ff_roi_name = roi_name.replace('-', '_').lower()
        label_path = os.path.join(labels_path, f"{ff_roi_name}.npy")
        np.save(label_path, label)
        

# List patients.
pat_dirs = os.listdir(DATA_SOURCE)
pat_paths = sorted([os.path.join(DATA_SOURCE, d) for d in pat_dirs])
if args.numpats:
    pat_paths = pat_paths[:args.numpats]
 
# For each patient.
for pat_path in tqdm(pat_paths):
    # Skip patient if alread processed.
    pat_proc_path = os.path.join(DATA_DEST, pat_path.split(os.sep)[-1])
    if not args.overwrite and already_converted(pat_proc_path):
        print("Already converted, skipping.")
        continue

    # Load dicoms.
    ct_dicoms = utils.load_ct_dicoms(pat_path)
    ct_dicoms_info_df = utils.get_ct_dicoms_info(ct_dicoms)
    ct_dicoms_info_summary_df = utils.get_ct_dicoms_info_summary(ct_dicoms_info_df)
    rtstruct_dicom = utils.load_rtstruct_dicom(pat_path)

    # Create placeholder for input data.
    input_shape = (ct_dicoms_info_summary_df['dim-x'][0], ct_dicoms_info_summary_df['dim-y'][0], ct_dicoms_info_summary_df['dim-z'][0])
    input_data = np.zeros(shape=input_shape)

    # Add slice data.
    print(input_data.shape)
    print(len(ct_dicoms))
    for dicom in ct_dicoms:
        offset_z =  dicom.ImagePositionPatient[2] - ct_dicoms_info_summary_df['offset-z'][0]
        pixel_z = int(offset_z / ct_dicoms_info_summary_df['res-z'][0])
        input_data[:, :, pixel_z] = utils.get_pixels(dicom)

    # Get label info.
    labels_data = get_labels(rtstruct_dicom, ct_dicoms_info_summary_df)

    # Save raw data.
    save_data(pat_proc_path, input_data, labels_data)

exit(0)



