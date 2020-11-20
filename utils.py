import numpy as np
import os
import pandas as pd
import pydicom as dicom

DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1')

def get_pat_path(pat_id):
    """
    Returns path by patient ID.
    pat_id: patient ID, e.g. 'HN1004'.
    """
    data_path = os.path.join(DATA_ROOT, 'raw')
    pat_path = os.path.join(data_path, pat_id)

    assert os.path.exists(pat_path), f"No such patient '{pat_id}'"

    return pat_path

def get_summary():
    """
    Returns the summary dataframe.
    """
    summary_path = os.path.join(DATA_ROOT, 'summary', 'summary.csv')
    summary_df = pd.read_csv(summary_path, index_col=0)

    return summary_df

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

def get_ct_info(dicoms):
    """
    Gets fields of interest from each dicom.
    dicoms: a list of CT dicoms.
    """
    info_cols = {
        'dim-x': np.uint16,
        'dim-y': np.uint16,
        'hu-min': 'float64',
        'hu-max': 'float64',
        'offset-x': 'float64',
        'offset-y': 'float64',
        'res-x': 'float64',
        'res-y': 'float64',
        'scale-int': 'float64',
        'scale-slope': 'float64'
    }
    info_df = pd.DataFrame(columns=np.sort(list(info_cols.keys())))

    # Default sorting: z-position.
    dicoms = sorted(dicoms, key=lambda d: d.ImagePositionPatient[2])

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

    return info_df

def get_rtstruct_info(rtstruct_dicom):
    """
    Returns information for the RTSTRUCT segmentation file.
    rtstruct_dicom: the RTSTRUCT dicom.
    """
    info_cols = {
        'roi-num': np.uint16,
        'roi-label': 'object'
    }
    info_df = pd.DataFrame(columns=np.sort(list(info_cols.keys())))

    rois = rtstruct_dicom.StructureSetROISequence
    rois = sorted(rois, key=lambda r: r.ROIName)

    for roi in rois:
        roi_info = {}

        # Add number.
        roi_info['roi-num'] = roi.ROINumber

        # Add label.
        roi_info['roi-label'] = roi.ROIName

        info_df = info_df.append(roi_info, ignore_index=True)

    # 'DataFrame.append' crushes the type so enforce at the end. See: 
    # https://stackoverflow.com/questions/22044766/adding-row-to-pandas-dataframe-changes-dtype
    info_df = info_df.astype(info_cols)

    return info_df