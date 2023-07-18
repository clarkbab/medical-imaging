from ast import literal_eval
import json
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import pydicom as dcm
from re import match
from time import time
from tqdm import tqdm
from typing import Any, Dict, Optional

from mymi import config
from mymi import logging
from mymi.utils import append_dataframe, append_row

CT_FROM_REGEXP = r'^__CT_FROM_(.*)__$'
INDEX_INDEX_COL = 'sop-id'
INDEX_COLS = {
    'dataset': str,
    'patient-id': str,
    'study-id': str,
    'modality': str,
    'series-id': str,
    'filepath': str,
    'mod-spec': object
}
ERRORS_INDEX_COLS = INDEX_COLS.copy()
ERRORS_INDEX_COLS['error'] = str

DEFAULT_POLICY = {
    'ct': {
        'slice': {
            'inconsistent-spacing': {
                'allow': False
            },
            'non-standard-orientation': {
                'allow': False
            }
        },
        'slices': {
            'inconsistent-position': {
                'allow': False
            },
            'inconsistent-spacing': {
                'allow': False
            }
        }
    },
    'duplicates': {
        'allow': False
    },
    'rtdose': {
        'no-ref-rtplan': {
            'allow': False
        }
    },
    'rtplan': {
        'no-ref-rtstruct': {
            'allow': False
        }
    },
    'rtstruct': {
        'no-ref-ct': {
            'allow': False
        }
    },
    'study': {
        'no-rtstruct': {
            'allow': False
        }
    }
}
  
def build_index(
    dataset: str,
    from_temp_index: bool = False) -> None:
    start = time()
    logging.arg_log('Building index', ('dataset',), (dataset,))

    # Load dataset path.
    dataset_path = os.path.join(config.directories.datasets, 'dicom', dataset) 

    # Determine index policy.
    policy_path = os.path.join(dataset_path, 'index-policy.json')
    if os.path.exists(policy_path):
        logging.info(f"Using custom policy at '{policy_path}'.")
        with open(policy_path, 'r') as f:
            policy = json.load(f)
    else:
        logging.info('Using default policy.')
        policy = DEFAULT_POLICY

    # Check '__CT_FROM_<dataset>__' tag.
    ct_from = None
    for f in os.listdir(dataset_path):
        m = match(CT_FROM_REGEXP, f)
        if m:
            ct_from = m.group(1)

    # Create index.
    if ct_from is None:
        # Create index from scratch.
        modalities = ('CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE')
        index_index = pd.Index(data=[], name=INDEX_INDEX_COL)
        index = pd.DataFrame(columns=INDEX_COLS.keys(), index=index_index)
    else:
        # Create index using 'ct_from' index as a starting point.
        logging.info(f"Using CT index from '{ct_from}'.")
        modalities = ('RTSTRUCT', 'RTPLAN', 'RTDOSE')

        # Load 'ct_from' index - can't use DICOMDataset API as it creates circular dependencies.
        filepath = os.path.join(config.directories.datasets, 'dicom', ct_from, 'index.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"Index for 'ct_from={ct_from}' dataset doesn't exist. Filepath: '{filepath}'.")
        index = pd.read_csv(filepath, dtype={ 'patient-id': str }, index_col=INDEX_INDEX_COL)
        index = index[index['modality'] == 'CT']
        index['mod-spec'] = index['mod-spec'].apply(lambda m: literal_eval(m))      # Convert str to dict.

    # Crawl folders to find all DICOM files.
    temp_filepath = os.path.join(config.directories.temp, f'{dataset}-index.csv')
    if from_temp_index:
        if os.path.exists(temp_filepath):
            logging.info(f"Loading saved index for dataset '{dataset}'.")
            index = pd.read_csv(temp_filepath, index_col=INDEX_INDEX_COL)
            index['mod-spec'] = index['mod-spec'].apply(lambda m: literal_eval(m))      # Convert str to dict.
        else:
            raise ValueError(f"Temporary index doesn't exist for dataset '{dataset}' at filepath '{temp_filepath}'.")
    else:
        data_path = os.path.join(dataset_path, 'data')
        if not os.path.exists(data_path):
            raise ValueError(f"No 'data' folder found for dataset '{dataset}'.")

        # Add all DICOM files.
        for root, _, files in tqdm(os.walk(data_path)):
            for f in files:
                # Check if DICOM file.
                filepath = os.path.join(root, f)
                try:
                    dicom = dcm.read_file(filepath, stop_before_pixels=True)
                except dcm.errors.InvalidDicomError:
                    continue

                # Get modality.
                modality = dicom.Modality
                if not modality in modalities:
                    continue

                # Get patient ID.
                pat_id = dicom.PatientID

                # Get study UID.
                study_id = dicom.StudyInstanceUID

                # Get series UID.
                series_id = dicom.SeriesInstanceUID

                # Get SOP UID.
                sop_id = dicom.SOPInstanceUID

                # Get modality-specific info.
                if modality == 'CT':
                    if not hasattr(dicom, 'ImageOrientationPatient'):
                        logging.error(f"No 'ImageOrientationPatient' found for CT dicom '{filepath}'.")
                        continue

                    mod_spec = {
                        'ImageOrientationPatient': dicom.ImageOrientationPatient,
                        'ImagePositionPatient': dicom.ImagePositionPatient,
                        'InstanceNumber': dicom.InstanceNumber,
                        'PixelSpacing': dicom.PixelSpacing
                    }
                elif modality == 'RTDOSE':
                    mod_spec = {
                        'RefRTPLANSOPInstanceUID': dicom.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                    }
                elif modality == 'RTPLAN':
                    mod_spec = {
                        'RefRTSTRUCTSOPInstanceUID': dicom.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                    }
                elif modality == 'RTSTRUCT':
                    mod_spec = {
                        'RefCTSeriesInstanceUID': dicom.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    }

                # Add index entry.
                data_index = sop_id
                data = {
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'study-id': study_id,
                    'modality': modality,
                    'series-id': series_id,
                    'filepath': filepath,
                    'mod-spec': mod_spec,
                }
                index = append_row(index, data, index=data_index)
    
        # Save index - in case something goes wrong later.
        index.to_csv(temp_filepath, index=True)

    # Create errors index.
    errors_index_index = pd.Index([], name=INDEX_INDEX_COL)
    errors_index = pd.DataFrame(columns=ERRORS_INDEX_COLS.keys(), index=errors_index_index)

    # Remove duplicates by 'SOPInstanceUID'.
    if not policy['duplicates']['allow']:
        logging.info(f"Removing duplicate DICOM files (by 'SOPInstanceUID').")

        dup_rows = index.index.duplicated()
        dup = index[dup_rows]
        dup['error'] = 'DUPLICATE'
        errors_index = append_dataframe(errors_index, dup)
        index = index[~dup_rows]

    # Check CT slices have standard orientation.
    if ct_from is None and not policy['ct']['slice']['non-standard-orientation']['allow']:
        logging.info(f"Removing CT DICOM files with rotated orientation (by 'ImageOrientationPatient').")

        ct = index[index['modality'] == 'CT']
        def standard_orientation(m: Dict) -> bool:
            orient = m['ImageOrientationPatient']
            return orient == [1, 0, 0, 0, 1, 0]
        stand_orient = ct['mod-spec'].apply(standard_orientation)
        nonstand_idx = stand_orient[~stand_orient].index
        nonstand = index.loc[nonstand_idx]
        nonstand['error'] = 'NON-STANDARD-ORIENTATION'
        errors_index = append_dataframe(errors_index, nonstand)
        index = index.drop(nonstand_idx)

    # Check CT slices have consistent x/y spacing.
    if ct_from is None and not policy['ct']['slice']['inconsistent-spacing']['allow']:
        logging.info(f"Removing CT DICOM files with inconsistent x/y spacing (by 'PixelSpacing').")

        ct = index[index['modality'] == 'CT']
        def consistent_xy_spacing(series: pd.Series) -> bool:
            pos = series.apply(lambda m: pd.Series(m['PixelSpacing']))
            pos = pos.drop_duplicates()
            return len(pos) == 1
        cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_spacing)
        incons_idx = cons_xy[~cons_xy].index
        incons = index.loc[incons_idx]
        incons['error'] = 'INCONSISTENT-SPACING-XY'
        errors_index = append_dataframe(errors_index, incons)
        index = index.drop(incons_idx)

    # Check CT slices have consistent x/y position.
    if ct_from is None and not policy['ct']['slices']['inconsistent-position']['allow']:
        logging.info(f"Removing CT DICOM files with inconsistent x/y position (by 'ImagePositionPatient').")

        ct = index[index['modality'] == 'CT']
        def consistent_xy_position(series: pd.Series) -> bool:
            pos = series.apply(lambda m: pd.Series(m['ImagePositionPatient'][:2]))
            pos = pos.drop_duplicates()
            return len(pos) == 1
        cons_xy = ct[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(consistent_xy_position)
        incons_idx = cons_xy[~cons_xy].index
        incons = index.loc[incons_idx]
        incons['error'] = 'INCONSISTENT-POSITION-XY'
        errors_index = append_dataframe(errors_index, incons)
        index = index.drop(incons_idx)

    # Check CT slices have consistent z spacing.
    if ct_from is None and not policy['ct']['slices']['inconsistent-spacing']['allow']:
        logging.info(f"Removing CT DICOM files with inconsistent z spacing (by 'ImagePositionPatient').")

        ct = index[index['modality'] == 'CT']
        def consistent_z_position(series: pd.Series) -> bool:
            z_locs = series.apply(lambda m: m['ImagePositionPatient'][2]).sort_values()
            z_diffs = z_locs.diff().dropna().round(3)
            z_diffs = z_diffs.drop_duplicates()
            return len(z_diffs) == 1
        cons_z = ct.groupby('series-id')['mod-spec'].transform(consistent_z_position)
        incons_idx = cons_z[~cons_z].index
        incons = index.loc[incons_idx]
        incons['error'] = 'INCONSISTENT-SPACING-Z'
        errors_index = append_dataframe(errors_index, incons)
        index = index.drop(incons_idx)

    # Load RTSTRUCT series without referenced CT series.
    ct_series = index[index['modality'] == 'CT']['series-id'].unique()
    rtstruct_series = index[index['modality'] == 'RTSTRUCT']
    ref_ct = rtstruct_series['mod-spec'].apply(lambda m: m['RefCTSeriesInstanceUID']).isin(ct_series)
    no_ref_ct_idx = ref_ct[~ref_ct].index
    no_ref_ct = index.loc[no_ref_ct_idx]

    if not policy['rtstruct']['no-ref-ct']['allow']:
        # Check that RTSTRUCT references CT series in index.
        logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID').")

        # Discard RTSTRUCTS with no referenced CT.
        no_ref_ct['error'] = 'NO-REF-CT'
        errors_index = append_dataframe(errors_index, no_ref_ct)
        index = index.drop(no_ref_ct_idx)

    else:
        # Add study's CT series count info to RTSTRUCT table.
        study_ct_series_count = index[index['modality'] == 'CT'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'ct-count' })
        no_ref_ct = no_ref_ct.reset_index().merge(study_ct_series_count, how='left', on='study-id').set_index(INDEX_INDEX_COL)

        if policy['rtstruct']['no-ref-ct']['only'] == 'at-least-one-ct':
            logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID') and with no CT series in the study.")

            # Discard RTSTRUCTs with no CT in study.
            no_ct_series_idx = no_ref_ct[no_ref_ct['ct-count'].isna()].index
            no_ct_series = index.loc[no_ct_series_idx]
            no_ct_series['error'] = 'NO-REF-CT:NO-CT-SERIES'
            errors_index = append_dataframe(errors_index, no_ct_series)
            index = index.drop(no_ct_series_idx)

        elif policy['rtstruct']['no-ref-ct']['only'] == 'single-ct':
            logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID') and with multiple or no CT series in the study.")

            # Discard RTSTRUCTs with no CT in study.
            no_ct_series_idx = no_ref_ct[no_ref_ct['ct-count'].isna()].index
            no_ct_series = index.loc[no_ct_series_idx]
            no_ct_series['error'] = 'NO-REF-CT:NO-CT-SERIES'
            errors_index = append_dataframe(errors_index, no_ct_series)
            index = index.drop(no_ct_series_idx)
            multiple_ct_series_idx = no_ref_ct[no_ref_ct['ct-count'] != 1].index
            multiple_ct_series = index.loc[multiple_ct_series_idx]
            multiple_ct_series['error'] = 'NO-REF-CT:MULTIPLE-CT-SERIES'
            errors_index = append_dataframe(errors_index, multiple_ct_series)
            index = index.drop(multiple_ct_series_idx)

    # Check that RTPLAN references RTSTRUCT SOP instance ID in index.
    rtstruct_sops = index[index['modality'] == 'RTSTRUCT'].index
    rtplan = index[index['modality'] == 'RTPLAN']
    ref_rtstruct = rtplan['mod-spec'].apply(lambda m: m['RefRTSTRUCTSOPInstanceUID']).isin(rtstruct_sops)
    no_ref_rtstruct_idx = ref_rtstruct[~ref_rtstruct].index
    no_ref_rtstruct = index.loc[no_ref_rtstruct_idx]

    if not policy['rtplan']['no-ref-rtstruct']['allow']:
        logging.info(f"Removing RTPLAN DICOM files without RTSTRUCT in index (by 'RefRTSTRUCTSOPInstanceUID').")

        # Discard RTPLANs without references RTSTRUCT.
        no_ref_rtstruct['error'] = 'NO-REF-RTSTRUCT'
        errors_index = append_dataframe(errors_index, no_ref_rtstruct)
        index = index.drop(no_ref_rtstruct_idx)

    else:
        # Add study's RSTRUCT series count info to RTPLAN table.
        study_rtstruct_series_count = index[index['modality'] == 'RTSTRUCT'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'rtstruct-count' })
        no_ref_rtstruct = no_ref_rtstruct.reset_index().merge(study_rtstruct_series_count, how='left', on='study-id').set_index(INDEX_INDEX_COL)

        if policy['rtplan']['no-ref-rtstruct']['only'] == ['at-least-one-rtstruct']:
            logging.info(f"Removing RTPLAN DICOM files without RTSTRUCT in index (by 'RefRTSTRUCTSOPInstanceUID') and with no RTSTRUCT in the study.")

            # Remove RTPLANs with no RTSTRUCT in study.
            no_rtstruct_series_idx = no_ref_rtstruct[no_ref_rtstruct['rtstruct-count'].isna()].index
            no_rtstruct_series = index.loc[no_rtstruct_series_idx]
            no_rtstruct_series['error'] = 'NO-REF-RTSTRUCT:NO-RTSTRUCT-SERIES'
            errors_index = append_dataframe(errors_index, no_rtstruct_series)
            index = index.drop(no_rtstruct_series_idx)

    # Check that RTDOSE references RTPLAN SOP in index.
    rtplan_sops = index[index['modality'] == 'RTPLAN'].index
    rtdose = index[index['modality'] == 'RTDOSE']
    ref_rtplan = rtdose['mod-spec'].apply(lambda m: m['RefRTPLANSOPInstanceUID']).isin(rtplan_sops)
    no_ref_rtplan_idx = ref_rtplan[~ref_rtplan].index
    no_ref_rtplan = index.loc[no_ref_rtplan_idx]

    # Check that RTDOSE references RTPLAN SOP instance in index.
    if not policy['rtdose']['no-ref-rtplan']['allow']:
        logging.info(f"Removing RTDOSE DICOM files without RTPLAN in index (by 'RefRTPLANSOPInstanceUID').")

        # Discard RTDOSEs with no referenced RTPLAN.
        no_ref_rtplan['error'] = 'NO-REF-RTPLAN'
        errors_index = append_dataframe(errors_index, no_ref_rtplan)
        index = index.drop(no_ref_rtplan_idx)

    else:
        # Add study's RTPLAN series count info to RTDOSE table.
        study_rtplan_series_count = index[index['modality'] == 'RTPLAN'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'rtplan-count' })
        no_ref_rtplan = no_ref_rtplan.reset_index().merge(study_rtplan_series_count, how='left', on='study-id').set_index(INDEX_INDEX_COL)

        if policy['rtdose']['no-ref-rtplan']['only'] == ['at-least-one-rtplan']:
            logging.info(f"Removing RTDOSE DICOM files without RTPLAN in index (by 'RefRTPLANSOPInstanceUID') and with no RTPLAN in the study.")

            # Remove RTDOSEs with no RTPLAN in the study.
            no_rtplan_series_idx = no_ref_rtplan[no_ref_rtplan['rtplan-count'].isna()].index
            no_rtplan_series = index.loc[no_rtplan_series_idx]
            no_rtplan_series['error'] = 'NO-REF-RTPLAN:NO-RTPLAN-SERIES'
            errors_index = append_dataframe(errors_index, no_rtplan_series)
            index = index.drop(no_rtplan_series_idx)

    # Check that study has RTSTRUCT series.
    if not policy['study']['no-rtstruct']['allow']:
        logging.info(f"Removing series without RTSTRUCT DICOM.")

        incl_rows = index.groupby('study-id')['modality'].transform(lambda s: 'RTSTRUCT' in s.unique())
        nonincl = index[~incl_rows]
        nonincl['error'] = 'STUDY-NO-RTSTRUCT'
        errors_index = append_dataframe(errors_index, nonincl)
        index = index[incl_rows]

    # Save index.
    if len(index) > 0:
        index = index.astype(INDEX_COLS)
    filepath = os.path.join(dataset_path, 'index.csv')
    index.to_csv(filepath, index=True)

    # Save errors index.
    if len(errors_index) > 0:
        errors_index = errors_index.astype(ERRORS_INDEX_COLS)
    filepath = os.path.join(dataset_path, 'index-errors.csv')
    errors_index.to_csv(filepath, index=True)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(dataset_path, f'__INDEXING_TIME_MINS_{mins}__')
    Path(filepath).touch()
