import ast
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pydicom as dcm
from re import match
from time import time
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.constants import *
from mymi import logging
from mymi.utils import *

from ...dataset import CT_FROM_REGEXP

filepath = os.path.join(os.path.dirname(__file__), 'default-policy.yaml')
DEFAULT_POLICY = load_yaml(filepath)

INDEX_COLS = {
    'dataset': str,
    'patient-id': str,
    'study-id': str,
    'study-date': str,
    'study-time': str,
    'series-id': str,
    'series-date': str,
    'series-time': str,
    'modality': str,
    'sop-id': str,
    'mod-spec': object,
    'filepath': str,
}
ERROR_INDEX_COLS = INDEX_COLS.copy()
ERROR_INDEX_COLS['error'] = str
 
def build_index(
    dataset: str,
    force_dicom_read: bool = False,
    n_crawl: Optional[int] = None,  # For testing purposes.
    recreate: bool = False,     # Just in case the index reaches a bad state.
    skip_crawl: bool = False) -> None:
    start = time()
    logging.info(f"Building index for dataset '{dataset}'.")

    # Load dataset path.
    dataset_path = os.path.join(config.directories.datasets, 'dicom', dataset) 

    # Check if indexes are open and therefore can't be overwritten.
    files = ['index.csv', 'index-errors.csv']
    filepaths = [os.path.join(dataset_path, f) for f in files]
    assert_can_write(filepaths)

    # Remove markers.
    files = os.listdir(dataset_path)
    for f in files:
        if f.startswith('__INDEXING_COMPLETE_'):
            os.remove(os.path.join(dataset_path, f))

    # Create or load policy.
    filepath = os.path.join(dataset_path, 'index-policy.yaml')
    if recreate or not os.path.exists(filepath):
        # Load custom policy.
        filepath = os.path.join(dataset_path, 'custom-policy.yaml')
        custom_policy = load_yaml(filepath) if os.path.exists(filepath) else None

        # Merge with default policy.
        policy = deep_merge(custom_policy, DEFAULT_POLICY) if custom_policy is not None else DEFAULT_POLICY
        filepath = os.path.join(dataset_path, 'index-policy.yaml')
        save_yaml(policy, filepath)
    else:
        policy = load_yaml(filepath)

    # Check '__CT_FROM_<dataset>__' tag.
    ct_from = None
    for f in os.listdir(dataset_path):
        m = match(CT_FROM_REGEXP, f)
        if m:
            ct_from = m.group(1)

    # Create or load index.
    modalities = list(get_args(DicomModality))
    filepath = os.path.join(dataset_path, 'index.csv')
    temp_filepath = os.path.join(config.directories.temp, f'{dataset}-index.csv')
    if skip_crawl:
        # Use temporary index (before policy filtering applied) - must have been saved by previous indexing.
        if os.path.exists(temp_filepath):
            logging.info(f"Skipping crawl of 'data/patients' folder, loading from temp index '{temp_filepath}'.")
            index = load_csv(temp_filepath, map_types=INDEX_COLS, parse_cols='mod-spec')
        else:
            raise ValueError(f"Temporary index file '{temp_filepath}' doesn't exist. Cannot skip crawl of 'data/patients' folder.")
    elif recreate or not os.path.exists(filepath):
        if ct_from is None:
            # Create new index.
            index = pd.DataFrame(columns=INDEX_COLS.keys())
        else:
            # Create index using 'ct_from' index as a starting point.
            logging.info(f"Using CT index from '{ct_from}'.")
            modalities.remove('ct')     # Don't look for new CT files.

            # Load 'ct_from' index - can't use DicomDataset API as it creates circular dependencies.
            filepath = os.path.join(config.directories.datasets, 'dicom', ct_from, 'index.csv')
            if not os.path.exists(filepath):
                raise ValueError(f"Index for 'ct_from={ct_from}' dataset doesn't exist. Filepath: '{filepath}'.")
            index = load_csv(filepath, filters={ 'modality': 'ct' }, map_types=INDEX_COLS, parse_cols='mod-spec')
    else:
        # Load existing index.
        index = load_csv(filepath, map_types=INDEX_COLS)

        # Remove files that are no longer present.
        for i, row in index.iterrows():
            filepath = os.path.join(dataset_path, 'data', 'patients', row['filepath'])
            if not os.path.exists(filepath):
                logging.warning(f"Removing index entry for '{filepath}' as it no longer exists.")
                index.drop(i, inplace=True)

    # Create or load index errors.
    # How does appending new entries to the index affect the error index?
    # We probably need to rebuild this from the index completely every time, as adding new entries
    # to the index could then make other files valid (e.g. a CT series that requires RTSTRUCT in the study).
    # Additionally, policy changes could make an invalid series valid.
    filepath = os.path.join(dataset_path, 'index-errors.csv')
    # if recreate or not os.path.exists(filepath):
    #     index_errors = pd.DataFrame(columns=ERROR_INDEX_COLS.keys())
    # else:
    #     index_errors = load_csv(filepath, map_types=ERROR_INDEX_COLS)
    index_errors = pd.DataFrame(columns=ERROR_INDEX_COLS.keys())

    # Crawl for new files.
    if not skip_crawl:
        # Crawl 'data/patients' folder for dicom files.
        data_path = os.path.join(dataset_path, 'data', 'patients')
        if not os.path.exists(data_path):
            raise ValueError(f"No 'data/patients' folder found for dataset '{dataset}'.")

        n_files = 0
        for root, _, files in tqdm(os.walk(data_path)):
            if n_crawl is not None and n_files >= n_crawl:
                logging.info(f"Reached limit of {n_crawl} files, stopping crawl.")
                break

            for f in files:
                if n_crawl is not None and n_files >= n_crawl:
                    break

                # Skip if file already added during previous indexing run.
                filepath = os.path.join(root, f)
                rel_filepath = filepath.replace(data_path, '').lstrip(os.sep)
                # Don't exclude 'error-index' files here, we need to check if these
                # files have become valid due to the inclusion of other series, or policy changes.
                if rel_filepath in index['filepath'].values:
                    continue

                # Check if valid dicom file.
                try:
                    dicom = dcm.dcmread(filepath, force=force_dicom_read, stop_before_pixels=True)
                except dcm.errors.InvalidDicomError:
                    continue

                # Get modality.
                modality = dicom.Modality.lower()
                if not modality in modalities:
                    continue

                # Get patient ID.
                pat_id = dicom.PatientID

                # Get study info.
                study_id = dicom.StudyInstanceUID
                if study_id in policy['study']['map-ids']:
                    study_id = policy['study']['map-ids'][study_id]
                    logging.info(f"Mapped study ID '{dicom.StudyInstanceUID}' to '{study_id}'.")
                study_date = dicom.StudyDate
                study_time = dicom.StudyTime

                # Get series info.
                series_id = dicom.SeriesInstanceUID
                series_date = dicom.SeriesDate if hasattr(dicom, 'SeriesDate') else ''
                series_time = dicom.SeriesTime if hasattr(dicom, 'SeriesTime') else ''

                # Get SOP UID.
                sop_id = dicom.SOPInstanceUID

                # Get modality-specific info.
                if modality == 'ct' or modality == 'mr':
                    if not hasattr(dicom, 'ImageOrientationPatient'):
                        logging.error(f"No 'ImageOrientationPatient' found for {modality} dicom '{filepath}'.")
                        continue

                    mod_spec = {
                        'ImageOrientationPatient': dicom.ImageOrientationPatient,
                        'ImagePositionPatient': dicom.ImagePositionPatient,
                        'InstanceNumber': dicom.InstanceNumber,
                        'PixelSpacing': dicom.PixelSpacing
                    }
                elif modality == 'rtdose':
                    # This key is conditionally required.
                    if hasattr(dicom, 'ReferencedRTPlanSequence') and len(dicom.ReferencedRTPlanSequence) > 0:
                        mod_spec = {
                            DICOM_RTDOSE_REF_RTPLAN_KEY: dicom.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                        }
                    else:
                        mod_spec = {}
                elif modality == 'rtplan':
                    mod_spec = {
                        DICOM_RTPLAN_REF_RTSTRUCT_KEY: dicom.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                    }
                elif modality == 'rtstruct':
                    mod_spec = {
                        DICOM_RTSTRUCT_REF_CT_KEY: dicom.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                    }

                # Add index entry.
                data = {
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'study-id': study_id,
                    'study-date': study_date,
                    'study-time': study_time,
                    'series-id': series_id,
                    'series-date': series_date,
                    'series-time': series_time,
                    'modality': modality,
                    'sop-id': sop_id,
                    'mod-spec': mod_spec,
                    'filepath': rel_filepath,
                }
                index = append_row(index, data)

                # Increment crawled files counter.
                n_files += 1
    
        # Save temporary index - before policy is applied.
        # Allows us to skip folder crawl step when iterating on policies.
        save_csv(index, temp_filepath)

        # Map 'mod-spec' column to literal.
        def map_mod_spec(m: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
            # Index could have both dict (from loaded existing index, recreate=False)
            # and string (from newly crawled files) mod-spec values.
            return ast.literal_eval(m) if isinstance(m, str) else m
        index['mod-spec'] = index['mod-spec'].apply(map_mod_spec)

    # Check that any files were found.
    if len(index) == 0:
        raise ValueError("No dicom files found!")

    if not policy['duplicates']['allow']:
        # Remove duplicates by 'SOPInstanceUID'.
        logging.info(f"Removing duplicate DICOM files (by 'SOPInstanceUID').")
        is_dup = index['sop-id'].duplicated()
        dup_rows = index[is_dup].copy()
        dup_rows['error'] = 'DUPLICATE-SOP-ID'
        index_errors = append_dataframe(index_errors, dup_rows)
        index = index[~is_dup].copy()

    if ct_from is None and not policy['ct']['slice']['non-standard-orientation']['allow']:
        # Remove CT slices with non-standard orientation.
        logging.info(f"Removing CT DICOM files with rotated orientation (by 'ImageOrientationPatient').")
        ct_rows = index[index['modality'] == 'ct']
        def has_standard_orientation(m: Dict) -> bool:
            orient = m['ImageOrientationPatient']
            return orient == [1, 0, 0, 0, 1, 0]
        is_standard = ct_rows['mod-spec'].apply(has_standard_orientation)
        non_standard_rows = ct_rows[~is_standard].copy()
        non_standard_rows['error'] = 'NON-STANDARD-ORIENTATION'
        index_errors = append_dataframe(index_errors, non_standard_rows)
        index = index[~index['sop-id'].isin(list(non_standard_rows['sop-id']))].copy()

    if ct_from is None and not policy['ct']['slice']['inconsistent-spacing']['allow']:
        # Remove CT slices with inconsistent x/y spacing.
        logging.info(f"Removing CT DICOM files with inconsistent x/y spacing (by 'PixelSpacing').")
        ct_rows = index[index['modality'] == 'ct']
        def has_consistent_xy_spacing(series: pd.Series) -> bool:
            pos = series.apply(lambda m: pd.Series(m['PixelSpacing']))
            pos = pos.drop_duplicates()
            return len(pos) == 1
        is_consistent = ct_rows[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(has_consistent_xy_spacing)
        incons_rows = ct_rows[~is_consistent].copy()
        incons_rows['error'] = 'INCONSISTENT-SPACING-XY'
        index_errors = append_dataframe(index_errors, incons_rows)
        index = index[~index['sop-id'].isin(list(incons_rows['sop-id']))].copy()

    if ct_from is None and not policy['ct']['slices']['inconsistent-position']['allow']:
        # Remove CT series with inconsistent x/y position.
        # Won't work with non-standard orientation.
        logging.info(f"Removing CT DICOM files with inconsistent x/y position (by 'ImagePositionPatient').")
        ct_rows = index[index['modality'] == 'ct']
        def has_consistent_xy_position(series: pd.Series) -> bool:
            pos = series.apply(lambda m: pd.Series(m['ImagePositionPatient'][:2]))
            pos = pos.drop_duplicates()
            return len(pos) == 1
        is_consistent = ct_rows[['series-id', 'mod-spec']].groupby('series-id')['mod-spec'].transform(has_consistent_xy_position)
        incons_rows = ct_rows[~is_consistent].copy()
        incons_rows['error'] = 'INCONSISTENT-POSITION-XY'
        index_errors = append_dataframe(index_errors, incons_rows)
        index = index[~index['sop-id'].isin(list(incons_rows['sop-id']))].copy()

    if ct_from is None and not policy['ct']['slices']['inconsistent-spacing']['allow']:
        # Remove CT series with inconsistent z-spacing.
        logging.info(f"Removing CT DICOM files with inconsistent z spacing (by 'ImagePositionPatient').")
        ct_rows = index[index['modality'] == 'ct']
        def has_consistent_z_position(series: pd.Series) -> bool:
            z_locs = series.apply(lambda m: m['ImagePositionPatient'][2]).sort_values()
            z_diffs = z_locs.diff().dropna().round(3)
            z_diffs = z_diffs.drop_duplicates()
            return len(z_diffs) == 1
        is_consistent = ct_rows.groupby('series-id')['mod-spec'].transform(has_consistent_z_position)
        incons_rows = ct_rows[~is_consistent].copy()
        incons_rows['error'] = 'INCONSISTENT-SPACING-Z'
        index_errors = append_dataframe(index_errors, incons_rows)
        index = index[~index['sop-id'].isin(list(incons_rows['sop-id']))].copy()

    # Remove RSTRUCT series based on policy regarding referenced CT series.
    # 'no-ref-ct': {
    #   'allow': True/False,
    #   'in-study': '1'/'>=1'
    # }
    ct_series_ids = list(index[index['modality'] == 'ct']['series-id'].unique())
    rtstruct_rows = index[index['modality'] == 'rtstruct']
    has_ref_ct = rtstruct_rows['mod-spec'].apply(lambda m: m['RefCTSeriesInstanceUID']).isin(ct_series_ids)
    no_ref_ct = rtstruct_rows[~has_ref_ct].copy()

    if not policy['rtstruct']['no-ref-ct']['allow']:
        # Remove RTSTRUCT series that have no reference CT series.

        # Check that RTSTRUCT references CT series in index.
        logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID').")

        # Discard RTSTRUCTS with no referenced CT.
        no_ref_ct['error'] = 'NO-REF-CT'
        index_errors = append_dataframe(index_errors, no_ref_ct)
        index = index[~index['sop-id'].isin(list(no_ref_ct['sop-id']))].copy()
    else:
        # Add study's CT series count info to RTSTRUCT table.
        study_ct_series_count = index[index['modality'] == 'ct'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'ct-count' })
        no_ref_ct = no_ref_ct.merge(study_ct_series_count, how='left', on='study-id')

        if policy['rtstruct']['no-ref-ct']['require-ct-in-study']:
            # Remove RTSTRUCT series that don't have any CTs in the study.
            logging.info(f"Removing RTSTRUCT DICOM files without CT in index (by 'RefCTSeriesInstanceUID') and with no CT series in the study.")

            # Discard RTSTRUCTs with no CT in study.
            no_ct_series = no_ref_ct[no_ref_ct['ct-count'].isna()].copy()
            no_ct_series['error'] = 'NO-REF-CT:IN-STUDY:>=1'
            index_errors = append_dataframe(index_errors, no_ct_series)
            index = index[~index['sop-id'].isin(list(no_ct_series['sop-id']))].copy()

    # Remove RTPLAN series based on policy regarding referenced RTSTRUCT series.
    # 'no-ref-rtstruct': {
    #   'allow': True/False,
    #   'in-study': '1'/'>=1'
    # }
    rtstruct_sop_ids = list(index[index['modality'] == 'rtstruct']['sop-id'])
    rtplan_rows = index[index['modality'] == 'rtplan']
    has_ref_rtstruct = rtplan_rows['mod-spec'].apply(lambda m: m['RefRTSTRUCTSOPInstanceUID']).isin(rtstruct_sop_ids)
    no_ref_rtstruct = rtplan_rows[~has_ref_rtstruct].copy()

    if not policy['rtplan']['no-ref-rtstruct']['allow']:
        # Discard RTPLANs without reference RTSTRUCTs.
        logging.info(f"Removing RTPLAN DICOM files without a referenced RTSTRUCT ('RefRTSTRUCTSOPInstanceUID') in the index.")
        no_ref_rtstruct['error'] = 'NO-REF-RTSTRUCT'
        index_errors = append_dataframe(index_errors, no_ref_rtstruct)
        index = index[~index['sop-id'].isin(list(no_ref_rtstruct['sop-id']))].copy()

    elif policy['rtplan']['no-ref-rtstruct']['require-rtstruct-in-study']:
        # No direct RTSTRUCT is required (referenced by RTPLAN), but there must be an RTSTRUCT in the study.
        logging.info(f"Removing RTPLAN DICOM files without an RTSTRUCT with the same study ID in the index.")

        # Add study's RSTRUCT series count info to RTPLAN table.
        study_rtstruct_series_count = index[index['modality'] == 'rtstruct'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'rtstruct-count' })
        no_ref_rtstruct = no_ref_rtstruct.merge(study_rtstruct_series_count, how='left', on='study-id')

        # Remove RTPLANs with no RTSTRUCT in study.
        no_rtstruct_series = no_ref_rtstruct[no_ref_rtstruct['rtstruct-count'].isna()].copy()
        no_rtstruct_series['error'] = 'NO-REF-RTSTRUCT:NO-RTSTRUCT-IN-STUDY'
        index_errors = append_dataframe(index_errors, no_rtstruct_series)
        index = index[~index['sop-id'].isin(list(no_rtstruct_series['sop-id']))].copy()

    # Remove RTDOSE series based on policy regarding referenced RTPLAN series.
    rtplan_sop_ids = list(index[index['modality'] == 'rtplan']['sop-id'])
    rtdose_rows = index[index['modality'] == 'rtdose']
    has_ref_rtplan = rtdose_rows['mod-spec'].apply(lambda m: DICOM_RTDOSE_REF_RTPLAN_KEY in m and m[DICOM_RTDOSE_REF_RTPLAN_KEY]).isin(rtplan_sop_ids)
    no_ref_rtplan = rtdose_rows[~has_ref_rtplan].copy()

    # Check that RTDOSE references RTPLAN SOP instance in index.
    if not policy['rtdose']['no-ref-rtplan']['allow']:
        # Discard RTDOSEs with no referenced RTPLAN.
        logging.info(f"Removing RTDOSE DICOM files without a referenced RTPLAN ('{DICOM_RTDOSE_REF_RTPLAN_KEY}') in the index.")
        no_ref_rtplan['error'] = 'NO-REF-RTPLAN'
        index_errors = append_dataframe(index_errors, no_ref_rtplan)
        index = index[~index['sop-id'].isin(list(no_ref_rtplan['sop-id']))].copy()

    elif policy['rtdose']['no-ref-rtplan']['require-rtplan-in-study']:
        # No direct RTPLAN is required (referenced by RTDOSE), but there must be an RTPLAN in the study.
        logging.info(f"Removing RTDOSE DICOM files without an RTPLAN with the same study ID in the index.")

        # Add study's RTPLAN series count info to RTDOSE table.
        study_rtplan_series_count = index[index['modality'] == 'rtplan'][['study-id', 'series-id']].drop_duplicates().groupby('study-id').count().rename(columns={ 'series-id': 'rtplan-count' })
        no_ref_rtplan = no_ref_rtplan.merge(study_rtplan_series_count, how='left', on='study-id')

        # Remove RTDOSEs with no RTPLAN in the study.
        no_rtplan_series = no_ref_rtplan[no_ref_rtplan['rtplan-count'].isna()].copy()
        no_rtplan_series['error'] = 'NO-REF-RTPLAN:NO-RTPLAN-IN-STUDY'
        index_errors = append_dataframe(index_errors, no_rtplan_series)
        index = index[~index['sop-id'].isin(list(no_rtplan_series['sop-id']))].copy()

    # Filter studies that don't have the required modalities.
    modalities = ['ct', 'mr', 'rtstruct', 'rtplan', 'rtdose']
    for m in modalities:
        if not policy['study'][f'no-{m}']['allow']:
            # Remove studies without RTSTRUCT series.
            logging.info(f"Removing studies without {m} dicoms.")
            has_mod = index.groupby('study-id')['modality'].transform(lambda s: m in s.unique())
            no_mod = index[~has_mod].copy()
            no_mod['error'] = f'STUDY-NO-{m.upper()}'
            index_errors = append_dataframe(index_errors, no_mod)
            index = index[~index['sop-id'].isin(list(no_mod['sop-id']))].copy()

    # Save index.
    if len(index) > 0:
        index = index.astype(INDEX_COLS)
    filepath = os.path.join(dataset_path, 'index.csv')
    save_csv(index, filepath)

    # Save errors index.
    if len(index_errors) > 0:
        index_errors = index_errors.astype(ERROR_INDEX_COLS)
    filepath = os.path.join(dataset_path, 'index-errors.csv')
    save_csv(index_errors, filepath)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(dataset_path, f'__INDEXING_COMPLETE_{mins}_MINS__')
    Path(filepath).touch()

def exists(dataset: str) -> bool:
    dataset_path = os.path.join(config.directories.datasets, 'dicom', dataset) 
    files = os.listdir(dataset_path)
    for f in files:
        if f.startswith('__INDEXING_COMPLETE_'):
            return True
    return False
