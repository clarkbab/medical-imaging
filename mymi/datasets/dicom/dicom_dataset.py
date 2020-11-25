import os
import pydicom as pdcm

DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'raw')

class DicomDataset:
    # TODO: Flesh out.
    @staticmethod
    def has_id(pat_id):
        return True

    @staticmethod
    def list_ct(pat_id):
        """
        returns: a list of CT dicoms.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(DATA_ROOT, pat_id)
        date_path = os.path.join(pat_path, os.listdir(pat_path)[0])
        dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

        # Find first folder containing CT scans.
        for p in dicom_paths:
            file_path = os.path.join(p, os.listdir(p)[0])
            dicom = pdcm.read_file(file_path)
            
            if dicom.Modality == 'CT':
                ct_dicoms = [pdcm.read_file(os.path.join(p, d)) for d in os.listdir(p)]
                ct_dicoms = sorted(ct_dicoms, key=lambda d: d.ImagePositionPatient[2])
                return ct_dicoms

        # TODO: raise an error.
        return None

    @staticmethod
    def list_patients():
        """
        returns: a list of patient IDs.
        """
        return sorted(os.listdir(DATA_ROOT))

    @staticmethod
    def get_rtstruct(pat_id):
        """
        returns: the RTSTRUCT dicom.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(DATA_ROOT, pat_id)
        date_path = os.path.join(pat_path, os.listdir(pat_path)[0])
        dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

        # Find first folder containing CT scans.
        for p in dicom_paths:
            file_path = os.path.join(p, os.listdir(p)[0])
            dicom = pdcm.read_file(file_path)
            
            if dicom.Modality == 'RTSTRUCT':
                return dicom

        # TODO: raise an error.
        return None

