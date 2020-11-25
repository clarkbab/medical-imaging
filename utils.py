import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom as dicom

COMP_PRECISION = 2
DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1')
SUMMARY_ROOT = os.path.join(DATA_ROOT, 'summaries')
CT_SUMMARY_PATH = os.path.join(SUMMARY_ROOT, 'ct_summary.csv')
RTSTRUCT_SUMMARY_PATH = os.path.join(SUMMARY_ROOT, 'rtstruct_summary.csv')

def get_pat_paths():
    """
    Returns a list of all patient paths.
    """
    data_path = os.path.join(DATA_ROOT, 'raw')
    pat_paths = [os.path.join(data_path, p) for p in os.listdir(data_path)]

    return pat_paths

def get_pat_path(pat_id):
    """
    Returns path by patient ID.
    pat_id: patient ID, e.g. 'HN1004'.
    """
    data_path = os.path.join(DATA_ROOT, 'raw')
    pat_path = os.path.join(data_path, pat_id)

    assert os.path.exists(pat_path), f"No such patient '{pat_id}'"

    return pat_path
    
def get_pat_proc_path(pat_id):
    """
    Returns path to processed data by patient ID.
    pat_id: patient ID, e.g. 'HN1004'.
    """
    data_path = os.path.join(DATA_ROOT, 'processed')
    pat_path = os.path.join(data_path, pat_id)

    assert os.path.exists(pat_path), f"No such patient '{pat_id}'"

    return pat_path

# Input data and labels.
def get_input(pat_id):
    pat_proc_path = get_pat_proc_path(pat_id) 
    input_path = os.path.join(pat_proc_path, 'input.npy')
    return np.load(input_path)

# Dicom loader. Use pattern DicomLoader.from_path(pat_path) or DicomLoader.from_id(pat_id)
# PatientDicomLoader.from_id(pat_id).load_ct()
# PatientDicomLoader.from_path(pat_path).load_rtstruct()

def load_ct_dicoms(pat_path):
    """
    Returns CT dicom objects for a patient.
    pat_path: a path to the patient.
    """
    # Get dated subfolder path.
    date_path = os.path.join(pat_path, os.listdir(pat_path)[0])

    # Find CT scan folder.
    dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

    for p in dicom_paths:
        file_path = os.path.join(p, os.listdir(p)[0])
        dcm = dicom.read_file(file_path)
        
        if dcm.Modality == 'CT':
            ct_dicoms = [dicom.read_file(os.path.join(p, d)) for d in os.listdir(p)]
            ct_dicoms = sorted(ct_dicoms, key=lambda d: d.ImagePositionPatient[2])
            return ct_dicoms

    return None

def load_rtstruct_dicom(pat_path):
    """
    Returns RTSTRUCT dicom object for a patient.
    pat_path: path to the patient.
    """
    # Get dated subfolder path.
    date_path = os.path.join(pat_path, os.listdir(pat_path)[0])

    # Find rtstruct dicom path.
    dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

    # Test each folder by checking first dicom modality.
    for p in dicom_paths:
        dicom_path = os.path.join(p, os.listdir(p)[0])
        dcm = dicom.read_file(dicom_path)
        
        if dcm.Modality == 'RTSTRUCT':
            return dcm

    return None

# VolumeBuilder.create_ct_input(ct_dicoms)


def get_pixels(dicom):
    """
    Converts CT dicom file to pixel data.
    dicom: a CT dicom.
    """
    # Pull pixel data.
    pixel_data = dicom.pixel_array

    # Convert to pixel space to align with CT data.
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope
    pixel_data = slope * pixel_data + intercept
    pixel_data = pixel_data.astype(np.int16)

    return pixel_data

# Plotting utilities.

def plot_ct_dicom(dicom):
    """
    Plots a CT scan from a given dicom.
    dicom: the CT dicom.
    """
    pixel_data = get_pixels(dicom)

    plt.figure(figsize=(8, 8))
    plt.imshow(pixel_data, cmap='gray')
    plt.show()

def plot_ct_segmentation(ct_dicom, rtstruct_dicom):
    """
    Plots a CT scan and overlaid contour.
    ct_dicom: the CT dicom.
    rtstruct_dicom: the RTSTRUCT dicom.
    """
    pixel_data = get_pixels(ct_dicom)
    ct_info = get_ct_info([ct_dicom])

# Summaries.
#
# from mymi.dicom_summary import DicomSummary, PatientDicomSummary
#
# For running an entire summary.
#
# DicomSummary.summarise()
# options:
# - num_patients (default='all) - how many patients to run.
# read/write cache.
#
# For inspecting patient info.
#
# PatientDicomSummary.from(pat_id) or .from(pat_path)
# PatientDicomSummary.from(pat_id).extract_ct_info()
# options:
# - num_scans (default='all') - how many scans to run.
# PatientDicomSummary.from(pat_id).summarise_ct_info()
# PatientDicomSummary.from(pat_id).extract_rtstruct_info()
#
# We have options:
# read_cache (default=True) - if results exist on disk read those, otherwise recalculate.
# write_cache (default=True) - writes results to disk for caching.

def get_full_ct_summary():
    """
    Returns the summary dataframe.
    """
    summary_df = pd.read_csv(CT_SUMMARY_PATH, index_col=0)

    return summary_df

def get_rtstruct_summary():
    """
    Returns the label summary dataframe.
    """
    summary_df = pd.read_csv(RTSTRUCT_SUMMARY_PATH, index_col=0)

    return summary_df

def get_ct_dicoms_info(dicoms=None, pat_id=None):
    """
    Gets fields of interest from each dicom.
    dicoms: a list of CT dicoms.
    """
    if dicoms is None:
        pat_path = get_pat_path(pat_id)
        dicoms = get_ct_dicoms(pat_path)

    info_cols = {
        'dim-x': np.uint16,
        'dim-y': np.uint16,
        'hu-min': 'float64',
        'hu-max': 'float64',
        'offset-x': 'float64',
        'offset-y': 'float64',
        'offset-z': 'float64',
        'res-x': 'float64',
        'res-y': 'float64',
        'scale-int': 'float64',
        'scale-slope': 'float64'
    }
    info_df = pd.DataFrame(columns=info_cols.keys())

    for dicom in dicoms:
        dicom_info = {}

        # Add dimensions.
        dicom_info['dim-x'] = dicom.pixel_array.shape[0]
        dicom_info['dim-y'] = dicom.pixel_array.shape[1]

        # Add (0, 0) pixel offset.
        dicom_info['offset-x'] = dicom.ImagePositionPatient[0]
        dicom_info['offset-y'] = dicom.ImagePositionPatient[1]
        dicom_info['offset-z'] = dicom.ImagePositionPatient[2]

        # Add resolution.
        dicom_info['res-x'] = dicom.PixelSpacing[0]
        dicom_info['res-y'] = dicom.PixelSpacing[1]

        # Add rescale values.
        dicom_info['scale-int'] = dicom.RescaleIntercept
        dicom_info['scale-slope'] = dicom.RescaleSlope

        # Add HU range.
        hu = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
        dicom_info['hu-min'] = np.min(hu)
        dicom_info['hu-max'] = np.max(hu)

        info_df = info_df.append(dicom_info, ignore_index=True)

    # 'DataFrame.append' crushes the type so enforce at the end. See: 
    # https://stackoverflow.com/questions/22044766/adding-row-to-pandas-dataframe-changes-dtype
    info_df = info_df.astype(info_cols)

    # Sort by offset-z.
    info_df = info_df.sort_values('offset-z').reset_index(drop=True)

    return info_df

def get_ct_dicoms_info_summary(ct_dicoms_info):
    """
    Returns a DataFrame with the CT summary info for this patient.
    ct_dicoms_info: a DataFrame containing a row for each CT dicom.
    """
    # Define dataframe structure.
    ct_summary_info_cols = {
        'dim-x': np.uint16,
        'dim-y': np.uint16,
        'dim-z': np.uint16,
        'hu-min': 'float64',
        'hu-max': 'float64',
        'num-empty': np.uint16,
        'offset-x': 'float64',
        'offset-y': 'float64',
        'offset-z': 'float64',
        'res-x': 'float64',
        'res-y': 'float64',
        'res-z': 'float64', 
        'scale-int': 'float64',
        'scale-slope': 'float64',
    }

    # Check for consistency among patient scans.
    assert len(ct_dicoms_info['dim-x'].unique()) == 1
    assert len(ct_dicoms_info['dim-y'].unique()) == 1
    assert len(ct_dicoms_info['offset-x'].unique()) == 1
    assert len(ct_dicoms_info['offset-y'].unique()) == 1
    assert len(ct_dicoms_info['res-x'].unique()) == 1
    assert len(ct_dicoms_info['res-y'].unique()) == 1
    assert len(ct_dicoms_info['scale-int'].unique()) == 1
    assert len(ct_dicoms_info['scale-slope'].unique()) == 1
    
    # Calculate res-z - this will be the smallest available diff.
    res_zs = np.sort([round(i, COMP_PRECISION) for i in np.diff(ct_dicoms_info['offset-z'])])
    res_z = res_zs[0]

    # Calculate fov-z and dim-z.
    fov_z = ct_dicoms_info['offset-z'].max() - ct_dicoms_info['offset-z'].min()
    dim_z = int(fov_z / res_z) + 1

    # Calculate number of empty slices.
    num_empty = dim_z - len(ct_dicoms_info)

    # Create dataframe.
    ct_summary_data = {
        'dim-x': [ct_dicoms_info['dim-x'][0]],
        'dim-y': [ct_dicoms_info['dim-y'][0]],
        'dim-z': [dim_z],
        'fov-x': [ct_dicoms_info['dim-x'][0] * ct_dicoms_info['res-x'][0]],
        'fov-y': [ct_dicoms_info['dim-y'][0] * ct_dicoms_info['res-y'][0]],
        'fov-z': [dim_z * res_z],
        'hu-min': [ct_dicoms_info['hu-min'].min()],
        'hu-max': [ct_dicoms_info['hu-max'].max()],
        'num-empty': [num_empty],
        'offset-x': [ct_dicoms_info['offset-x'][0]],
        'offset-y': [ct_dicoms_info['offset-y'][0]],
        'offset-z': [ct_dicoms_info['offset-z'][0]],
        'res-x': [ct_dicoms_info['res-x'][0]],
        'res-y': [ct_dicoms_info['res-y'][0]],
        'res-z': [res_z],
        'scale-int': [ct_dicoms_info['scale-int'][0]],
        'scale-slope': [ct_dicoms_info['scale-slope'][0]],
    }
    ct_summary_df = pd.DataFrame(ct_summary_data, columns=ct_summary_info_cols.keys())

    return ct_summary_df

def get_rtstruct_info(rtstruct_dicom):
    """
    Returns information for the RTSTRUCT segmentation file.
    rtstruct_dicom: the RTSTRUCT dicom.
    """
    info_cols = {
        'roi-label': 'object'
    }
    info_df = pd.DataFrame(columns=np.sort(list(info_cols.keys())))

    rois = rtstruct_dicom.StructureSetROISequence
    rois = sorted(rois, key=lambda r: r.ROIName)

    for roi in rois:
        roi_info = {}

        # Add label.
        roi_info['roi-label'] = roi.ROIName

        info_df = info_df.append(roi_info, ignore_index=True)

    # 'DataFrame.append' crushes the type so enforce at the end. See: 
    # https://stackoverflow.com/questions/22044766/adding-row-to-pandas-dataframe-changes-dtype
    info_df = info_df.astype(info_cols)

    return info_df