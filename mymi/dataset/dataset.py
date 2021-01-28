import numpy as np
import os
import pandas as pd
import pydicom as pdcm
from tqdm import tqdm

from mymi import cache
from mymi import dataset

Z_SPACING_ROUND_DP = 2

class Dataset:
    @classmethod
    def data_dir(cls):
        raise NotImplementedError("Method 'data_dir' not implemented in subclass.")

    @classmethod
    def clinical_data(cls):
        raise NotImplementedError("Method 'clinical_data' not implemented in subclass.")
    
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
        pat_path = os.path.join(cls.data_dir(), pat_id)
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
        return sorted(os.listdir(cls.data_dir()))

    @classmethod
    def get_rtstruct(cls, pat_id):
        """
        returns: the RTSTRUCT dicom.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(cls.data_dir(), pat_id)
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

    @classmethod
    def summary(cls, **kwargs):
        """
        returns: a dataframe containing rows of patient summaries.
        kwargs:
            num_pats: the number of patients to summarise.
            pat_id: a string or list of patient IDs.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'summary',
            'kwargs': kwargs
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
                
        # Define table structure.
        cols = {
            'age': np.uint16,
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'num-missing': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'pat-id': 'object',
            'res-x': np.uint16,
            'res-y': np.uint16,
            'res-z': np.uint16,
            'roi-num': np.uint16,
            'sex': 'object',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
            'spacing-z': 'float64',
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        pat_ids = dataset.list_patients()

        # Retain only patients specified.
        if 'pat_id' in kwargs:
            if isinstance(kwargs['pat_id'], str):
                assert kwargs['pat_id'] in pat_ids
                pat_ids = [kwargs['pat_id']]
            else:
                for id in kwargs['pat_id']:
                    assert id in pat_ids
                pat_ids = kwargs['pat_id']

        # Run on subset of patients.
        if 'num_pats' in kwargs and kwargs['num_pats'] != 'all':
            assert isinstance(kwargs['num_pats'], int)
            pat_ids = pat_ids[:kwargs['num_pats']]

        # Add patient info.
        for pat_id in tqdm(pat_ids):
            patient_df = dataset.patient_summary(pat_id)
            patient_df['pat-id'] = pat_id
            df = df.append(patient_df)

        # Set column type.
        df = df.astype(cols)
        
        # Set index.
        df = df.set_index('pat-id')

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def region_count(cls, **kwargs):
        """
        returns: a dataframe containing regions and num patients with region.
        kwargs:
            num_pats: the number of patients to summarise.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'region_count',
            'kwargs': kwargs
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')

        # Define table structure.
        cols = {
            'num-patients': np.uint16,
            'region': 'object'
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        pat_ids = dataset.list_patients()

        # Run on subset of patients.
        if 'num_pats' in kwargs and kwargs['num_pats'] != 'all':
            assert isinstance(kwargs['num_pats'], int)
            pat_ids = pat_ids[:kwargs['num_pats']]

        for pat_id in tqdm(pat_ids):
            # Get RTSTRUCT info.
            region_info_df = cls.patient_regions(pat_id)

            # Add label counts.
            region_info_df['num-patients'] = 1
            df = df.merge(region_info_df, how='outer', on='region')
            df['num-patients'] = (df['num-patients_x'].fillna(0) + df['num-patients_y'].fillna(0)).astype(np.uint16)
            df = df.drop(['num-patients_x', 'num-patients_y'], axis=1)

        # Sort by 'roi-label'.
        df = df.sort_values('region').reset_index(drop=True)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def regions(cls, **kwargs):
        """
        returns: a dataframe linking patients to contoured regions.
        kwargs:
            num_pats: number of patients to summarise.
            pat_id: a string or list of patient IDs.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'regions',
            'kwargs': kwargs
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
                
        # Define table structure.
        cols = {
            'patient-id': 'object',
            'region': 'object'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pat_ids = dataset.list_patients()

        # Retain only patients specified.
        if 'pat_id' in kwargs:
            if isinstance(kwargs['pat_id'], str):
                assert kwargs['pat_id'] in pat_ids
                pat_ids = [kwargs['pat_id']]
            else:
                for id in kwargs['pat_id']:
                    assert id in pat_ids
                pat_ids = kwargs['pat_id']

        # Run on subset of patients.
        if 'num_pats' in kwargs and kwargs['num_pats'] != 'all':
            assert isinstance(kwargs['num_pats'], int)
            pat_ids = pat_ids[:kwargs['num_pats']]

        for pat_id in tqdm(pat_ids):
            # Get rtstruct info.
            region_info_df = cls.patient_regions(pat_id)

            # Add rows.
            for _, row in region_info_df.iterrows():
                data = {
                    'patient-id': pat_id,
                    'region': row['region']
                }
                df = df.append(data, ignore_index=True)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def patient_regions(cls, *args):
        """
        returns: dataframe with row for each region.
        args:
            pat_id: the patient ID.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'patient_regions',
            'args': args
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
        
        # Define table structure.
        cols = {
            'region': 'object'
        }
        df = pd.DataFrame(columns=cols.keys())

        assert len(args) == 1, 'No patient ID passed'
        pat_id = args[0]
        rois = dataset.get_rtstruct(pat_id).StructureSetROISequence
        
        # Add info for each region-of-interest.
        for roi in rois:
            data = {
                'region': roi.ROIName
            }
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by label.
        df = df.sort_values('region').reset_index(drop=True)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def ct(cls, **kwargs):
        """
        returns: a dataframe linking patients to contoured regions.
        kwargs:
            num_pats: number of patients to summarise.
            pat_id: a string or list of patient IDs.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'ct',
            'kwargs': kwargs
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
                
        # Define dataframe structure.
        cols = {
            'patient-id': 'object',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'res-x': np.uint16,
            'res-y': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pat_ids = dataset.list_patients()

        # Retain only patients specified.
        if 'pat_id' in kwargs:
            if isinstance(kwargs['pat_id'], str):
                assert kwargs['pat_id'] in pat_ids
                pat_ids = [kwargs['pat_id']]
            else:
                for id in kwargs['pat_id']:
                    assert id in pat_ids
                pat_ids = kwargs['pat_id']

        # Run on subset of patients.
        if 'num_pats' in kwargs and kwargs['num_pats'] != 'all':
            assert isinstance(kwargs['num_pats'], int)
            pat_ids = pat_ids[:kwargs['num_pats']]

        for pat_id in tqdm(pat_ids):
            # Get rtstruct info.
            ct_df = cls.patient_ct(pat_id)

            # Add rows.
            for _, row in ct_df.iterrows():
                data = {
                    'patient-id': pat_id,
                    'hu-min': row['hu-min'],
                    'hu-max': row['hu-max'],
                    'offset-x': row['offset-x'],
                    'offset-y': row['offset-y'],
                    'offset-z': row['offset-z'],
                    'res-x': row['res-x'],
                    'res-y': row['res-y'],
                    'scale-int': row['scale-int'],
                    'scale-slope': row['scale-slope'],
                    'spacing-x': row['spacing-x'],
                    'spacing-y': row['spacing-y']
                }
                df = df.append(data, ignore_index=True)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def patient_ct(cls, *args):
        """
        returns: dataframe with rows containing CT info.
        args:
            pat_id: the patient ID.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'patient_ct',
            'args': args
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
            
        # Define dataframe structure.
        cols = {
            'hu-min': 'float64',
            'hu-max': 'float64',
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'res-x': np.uint16,
            'res-y': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
        }
        df = pd.DataFrame(columns=cols.keys())

        assert len(args) == 1
        pat_id = args[0]
        ct_dicoms = dataset.list_ct(pat_id)
        
        # Add info.
        for ct_dicom in ct_dicoms:
            # Perform scaling from stored values to HU.
            hus = ct_dicom.pixel_array * ct_dicom.RescaleSlope + ct_dicom.RescaleIntercept

            data = {
               'hu-min': hus.min(),
               'hu-max': hus.max(),
               'offset-x': ct_dicom.ImagePositionPatient[0], 
               'offset-y': ct_dicom.ImagePositionPatient[1], 
               'offset-z': ct_dicom.ImagePositionPatient[2], 
               'res-x': ct_dicom.pixel_array.shape[1],  # Pixel array is stored (y, x) for plotting.
               'res-y': ct_dicom.pixel_array.shape[0],
               'scale-int': ct_dicom.RescaleIntercept,
               'scale-slope': ct_dicom.RescaleSlope,
               'spacing-x': ct_dicom.PixelSpacing[0],
               'spacing-y': ct_dicom.PixelSpacing[1]
            }
            df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        # Sort by 'offset-z'.
        df = df.sort_values('offset-z').reset_index(drop=True)

        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df

    @classmethod
    def patient_summary(cls, *args):
        """
        returns: dataframe with single row summary of CT images.
        args:
            pat_id: the patient ID.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset',
            'method': 'patient_summary',
            'args': args
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')

        # Define table structure.
        cols = {
            'age': np.uint16,
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'num-missing': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'res-x': np.uint16,
            'res-y': np.uint16,
            'res-z': np.uint16,
            'roi-num': np.uint16,
            'sex': 'object',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
            'spacing-z': 'float64',
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Get patient scan info.
        assert len(args) == 1
        pat_id = args[0]
        ct_df = cls.patient_ct(pat_id)

        # Check for consistency among scans.
        assert len(ct_df['res-x'].unique()) == 1
        assert len(ct_df['res-y'].unique()) == 1
        assert len(ct_df['offset-x'].unique()) == 1
        assert len(ct_df['offset-y'].unique()) == 1
        assert len(ct_df['spacing-x'].unique()) == 1
        assert len(ct_df['spacing-y'].unique()) == 1
        assert len(ct_df['scale-int'].unique()) == 1
        assert len(ct_df['scale-slope'].unique()) == 1

        # Calculate spacing-z - this will be the smallest available diff.
        spacings_z = np.sort([round(i, Z_SPACING_ROUND_DP) for i in np.diff(ct_df['offset-z'])])
        spacing_z = spacings_z[0]

        # Calculate fov-z and res-z.
        fov_z = ct_df['offset-z'].max() - ct_df['offset-z'].min()
        res_z = int(round(fov_z / spacing_z, 0) + 1)

        # Calculate number of empty slices.
        num_slices = len(ct_df)
        num_missing = res_z - num_slices

        # Get patient RTSTRUCT info.
        region_df = cls.patient_regions(pat_id)

        # Load clinical data.
        clinical_df = cls.clinical_data()

        # Add table row.
        data = {
            'age': clinical_df.loc[pat_id]['age_at_diagnosis'], 
            'fov-x': ct_df['res-x'][0] * ct_df['spacing-x'][0],
            'fov-y': ct_df['res-y'][0] * ct_df['spacing-y'][0],
            'fov-z': res_z * spacing_z,
            'hu-min': ct_df['hu-min'].min(),
            'hu-max': ct_df['hu-max'].max(),
            'num-missing': num_missing,
            'offset-x': ct_df['offset-x'][0],
            'offset-y': ct_df['offset-y'][0],
            'offset-z': ct_df['offset-z'][0],
            'res-x': ct_df['res-x'][0],
            'res-y': ct_df['res-y'][0],
            'res-z': res_z,
            'roi-num': len(region_df),
            'sex': clinical_df.loc[pat_id]['biological_sex'], 
            'spacing-x': ct_df['spacing-x'][0],
            'spacing-y': ct_df['spacing-y'][0],
            'spacing-z': spacing_z, 
            'scale-int': ct_df['scale-int'][0],
            'scale-slope': ct_df['scale-slope'][0],
        }
        df = df.append(data, ignore_index=True)

        # Set type.
        df = df.astype(cols)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, df, 'dataframe')

        return df
