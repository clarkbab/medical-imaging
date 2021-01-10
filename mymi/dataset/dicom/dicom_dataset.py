import os
import pydicom as pdcm

class DicomDataset:
    @classmethod
    def root_dir(cls):
        raise NotImplementedError("Method 'root_dir' not implemented in overriding class.")
    
    @classmethod
    def has_id(cls, pat_id):
        return True

    @classmethod
    def list_ct(cls, pat_id):
        """
        returns: a list of CT dicoms.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(cls.root_dir(), pat_id)
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

    @classmethod
    def list_patients(cls):
        """
        returns: a list of patient IDs.
        """
        return sorted(os.listdir(cls.root_dir()))

    @classmethod
    def get_rtstruct(cls, pat_id):
        """
        returns: the RTSTRUCT dicom.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(cls.root_dir(), pat_id)
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
