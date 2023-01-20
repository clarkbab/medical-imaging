from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
from typing import List, Literal, Tuple, Union

from mymi import config
from mymi import logging
from mymi import types
from mymi.utils import append_row, load_csv, save_csv

# To download images from PACS, your desktop must have a static IP configured
# and a record should be created in PACS admin to allow your images to be downloaded
# to this IP.
PACS_AE_TITLE = 'DCMTK_A40942'      # 'AE' (application entity) field in PACS admin.
PACS_DT_FORMAT = '%Y%m%d'
PACS_PORT = 11112                   # Incoming port - on our desktop.
PACS_REMOTE_AE_TITLE = 'pacsFIR'    # AE (application) entity title of remote PACS server.
PACS_REMOTE_IP = '10.126.42.16'
PACS_REMOTE_PORT = 2104 

RTDOSE_DELAY_DAYS = 7

def download_dicoms(
    date_format: str = '%d/%m/%Y',
    sep: str = '\t') -> None:
    # Load patient file.
    df = load_csv('patient-specific-models', 'data', 'urn.txt', sep=sep)

    # Create error table.
    cols = {
        'patient-id': str,
        'study-date': datetime,
        'parent-study': str,
        'parent-modality': str,
        'parent-id': str,
        'error': str
    }
    error_df = pd.DataFrame(columns=cols.keys())

    # Download files for each patient/date.
    for _, (pat_id, study_date) in df.iterrows():
        study_date = datetime.strptime(study_date, date_format)
        error_df = download_patient_dicoms(pat_id, study_date, error_df)

    # Save error file.
    save_csv(error_df, 'patient-specific-models', 'data', 'errors.csv', overwrite=True)

def download_patient_dicoms(
    pat_id: types.PatientID,
    study_date: datetime,
    error_df: pd.DataFrame) -> None:
    logging.arg_log("Downloading patient DICOMS", ('pat_id', 'study_date'), (pat_id, study_date))

    # Download RTDOSE files. Allow 1 week for creation of these after CT creation.
    study_date_start = datetime.strftime(study_date, PACS_DT_FORMAT)
    study_date_end = datetime.strftime(study_date + relativedelta(days=RTDOSE_DELAY_DAYS), PACS_DT_FORMAT)
    study_date = f'{study_date_start}-{study_date_end}'
    query_results = download_patient_rtdose_dicoms(pat_id, study_date)

    if len(query_results) == 0:
        # Add error entry.
        data = {
            'patient-id': pat_id,
            'study-date': study_date,
            'error': 'NO-RTDOSE'
        }
        error_df = append_row(error_df, data)
    
    # Download RTPLAN for each RTDOSE file.
    for study_id, _, rtdose_sop_id in query_results:
        filepath = image_filepath(pat_id, study_id, 'RTDOSE', rtdose_sop_id)
        rtdose = pydicom.read_file(filepath)
        query_results = download_patient_rtplan_dicoms(pat_id, rtdose)
        
        if len(query_results) == 0:
            # Add error entry.
            data = {
                'patient-id': pat_id,
                'study-date': study_date,
                'parent-study': study_id,
                'parent-modality': 'RTDOSE',
                'parent-id': rtdose_sop_id,
                'error': 'NO-RTPLAN'
            }
            error_df = append_row(error_df, data)

        # Download RTSTRUCT for each RTPLAN file.
        for _, _, rtplan_sop_id in query_results:
            filepath = image_filepath(pat_id, study_id, 'RTPLAN', rtplan_sop_id)
            rtplan = pydicom.read_file(filepath)
            query_results = download_patient_rtstruct_dicoms(pat_id, rtplan)

            if len(query_results) == 0:
                # Add error entry.
                data = {
                    'patient-id': pat_id,
                    'study-date': study_date,
                    'parent-study': study_id,
                    'parent-modality': 'RTPLAN',
                    'parent-id': rtplan_sop_id,
                    'error': 'NO-RTSTRUCT'
                }
                error_df = append_row(error_df, data)

            # Download CT series for each RTSTRUCT file.
            for _, _, rtstruct_sop_id in query_results:
                filepath = image_filepath(pat_id, study_id, 'RTSTRUCT', rtstruct_sop_id)
                rtstruct = pydicom.read_file(filepath)
                query_results = download_patient_ct_dicoms(pat_id, rtstruct)

                if len(query_results) == 0:
                    # Add error entry.
                    data = {
                        'patient-id': pat_id,
                        'study-date': study_date,
                        'parent-study': study_id,
                        'parent-modality': 'RTSTRUCT',
                        'parent-id': rtstruct_sop_id,
                        'error': 'NO-CT'
                    }
                    error_df = append_row(error_df, data)

    return error_df

def download_patient_rtdose_dicoms(
    pat_id: types.PatientID,
    study_date: str) -> List[Tuple[str, str]]:
    logging.arg_log("Downloading patient RTDOSE dicoms", ('pat_id', 'study_date'), (pat_id, study_date))

    # Search for RTDOSE files by date.
    query_results = queryPACS(pat_id, 'IMAGE', 'RTDOSE', study_date=study_date)
    print(query_results)

    for study_id, series_id, sop_id in query_results:
        # Create RTDOSE folder.
        folder = series_folder(pat_id, study_id, 'RTDOSE')
        os.makedirs(folder, exist_ok=True)

        # Download RTDOSE file.
        command = f"""
powershell movescu --verbose --study --key QueryRetrieveLevel=IMAGE --key PatientID={pat_id} --key StudyInstanceUID='{study_id}' \
--key SeriesInstanceUID='{series_id}' --key SOPInstanceUID='{sop_id}' --aetitle {PACS_AE_TITLE} --call {PACS_REMOTE_AE_TITLE} {PACS_REMOTE_IP} \
{PACS_REMOTE_PORT} --move {PACS_AE_TITLE} --output-directory '{folder}' --port {PACS_PORT}
        """
        logging.info(command)
        os.system(command)

        # Raise error if RTDOSE wasn't downloaded.
        filepath = image_filepath(pat_id, study_id, 'RTDOSE', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'RTDOSE', sop_id)

    return query_results

def download_patient_rtplan_dicoms(
    pat_id: types.PatientID,
    rtdose: Dataset) -> None:
    logging.arg_log("Downloading patient RTPLAN dicoms", ('pat_id', 'rtdose'), (pat_id, rtdose.SOPInstanceUID))

    # Get referenced RTPLAN ID.
    rtplan_id = rtdose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID

    # Query for RTPLAN dicoms.
    query_results = queryPACS(pat_id, 'IMAGE', 'RTPLAN', sop_id=rtplan_id) 

    for study_id, series_id, sop_id in query_results:
        # Create RTPLAN folder.
        folder = series_folder(pat_id, study_id, 'RTPLAN')
        os.makedirs(folder, exist_ok=True)

        # Download RTPLAN file.
        command = f"""
powershell movescu --verbose --study --key QueryRetrieveLevel=IMAGE --key PatientID={pat_id} --key StudyInstanceUID='{study_id}' \
--key SeriesInstanceUID='{series_id}' --key SOPInstanceUID='{sop_id}' --aetitle {PACS_AE_TITLE} --call {PACS_REMOTE_AE_TITLE} {PACS_REMOTE_IP} \
{PACS_REMOTE_PORT} --move {PACS_AE_TITLE} --output-directory '{folder}' --port {PACS_PORT}
        """
        logging.info(command)
        os.system(command)

        # Raise error if RTPLAN wasn't downloaded.
        filepath = image_filepath(pat_id, study_id, 'RTPLAN', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'RTPLAN', sop_id)

    return query_results

def download_patient_rtstruct_dicoms(
    pat_id: types.PatientID,
    rtplan: Dataset) -> None:
    logging.arg_log("Downloading patient RTSTRUCT dicoms", ('pat_id', 'rtplan'), (pat_id, rtplan.SOPInstanceUID))

    # Get referenced RTSTRUCT ID.
    rtstruct_id = rtplan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

    # Query for RTSTRUCT dicoms.
    query_results = queryPACS(pat_id, 'IMAGE', 'RTSTRUCT', sop_id=rtstruct_id) 

    for study_id, series_id, sop_id in query_results:
        # Create RTSTRUCT folder.
        folder = series_folder(pat_id, study_id, 'RTSTRUCT')
        os.makedirs(folder, exist_ok=True)

        # Download RTSTRUCT file.
        command = f"""
powershell movescu --verbose --study --key QueryRetrieveLevel=IMAGE --key PatientID={pat_id} --key StudyInstanceUID='{study_id}' \
--key SeriesInstanceUID='{series_id}' --key SOPInstanceUID='{sop_id}' --aetitle {PACS_AE_TITLE} --call {PACS_REMOTE_AE_TITLE} {PACS_REMOTE_IP} \
{PACS_REMOTE_PORT} --move {PACS_AE_TITLE} --output-directory '{folder}' --port {PACS_PORT}
        """
        logging.info(command)
        os.system(command)

        # Raise error if RTSTRUCT wasn't downloaded.
        filepath = image_filepath(pat_id, study_id, 'RTSTRUCT', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'RTSTRUCT', sop_id)

    return query_results

def download_patient_ct_dicoms(
    pat_id: types.PatientID,
    rtstruct: Dataset) -> List[Tuple[str, str]]:
    logging.arg_log("Downloading patient CT dicoms", ('pat_id', 'rtstruct'), (pat_id, rtstruct.SOPInstanceUID))

    # Get referenced CT series ID.
    ct_series_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID

    # Query for CT dicoms.
    query_results = queryPACS(pat_id, 'SERIES', 'CT', series_id=ct_series_id)
    print(query_results)

    for study_id, series_id in query_results:
        # Create CT folder.
        folder = series_folder(pat_id, study_id, 'CT')
        os.makedirs(folder, exist_ok=True)

        # Download CT files.
        command = f"""
powershell movescu --verbose --study --key QueryRetrieveLevel=IMAGE --key PatientID={pat_id} --key StudyInstanceUID='{study_id}' \
--key SeriesInstanceUID='{series_id}' --aetitle {PACS_AE_TITLE} --call {PACS_REMOTE_AE_TITLE} {PACS_REMOTE_IP} \
{PACS_REMOTE_PORT} --move {PACS_AE_TITLE} --output-directory '{folder}' --port {PACS_PORT}
        """
        logging.info(command)
        os.system(command)

        # Raise error if CT wasn't downloaded.
        folder = series_folder(pat_id, study_id, 'CT')
        files = os.listdir(folder)
        if len(files) == 0:
            raise_error(pat_id, study_id, 'CT', series_id)

    return query_results

def queryPACS(
    pat_id: types.PatientID,
    level: Literal['IMAGE', 'SERIES'],
    modality: Literal['CT', 'RTDOSE', 'RTPLAN', 'RTSTRUCT'],
    series_id: str = '*',
    sop_id: str = '*',
    study_date: str = '*') -> Union[List[Tuple[str, str]], List[Tuple[str, str]]]:

    # Connect with remote server.
    ae = AE()
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
    assoc = ae.associate(PACS_REMOTE_IP, PACS_REMOTE_PORT, ae_title=PACS_REMOTE_AE_TITLE)

    # Perform query.
    results = []
    if assoc.is_established:
        # Create query.
        ds = Dataset()
        ds.PatientID = str(pat_id)
        ds.QueryRetrieveLevel = level
        ds.Modality = modality
        ds.StudyDate = study_date
        ds.SeriesInstanceUID = series_id
        ds.StudyInstanceUID = '*'
        if level == 'IMAGE':
            ds.SOPInstanceUID = sop_id

        print('=== query ===')
        print(ds)

        # Use the C-FIND service to send the identifier
        responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)

        for (status, identifier) in responses:
            if status:
                #print('C-FIND query status: 0x{0:04x}'.format(status.Status))

                # If the status is 'Pending' then identifier is the C-FIND response
                if status.Status in (0xFF00, 0xFF01):
                    print('=== response ===')
                    print(identifier)
                    result = [identifier.StudyInstanceUID, identifier.SeriesInstanceUID]
                    if level == 'IMAGE':
                        result.append(identifier.SOPInstanceUID)
                    results.append(result)
            else:
                print('Connection timed out, was aborted or received invalid response')

         # Release the association
        assoc.release()
    else:
        print('Association rejected, aborted or never connected')

    return results

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, '%Y%m%d')

def study_folder(
    pat_id: types.PatientID,
    study_id: str):
    return os.path.join(config.directories.files, 'patient-specific-models', 'data', 'dcmFiles', str(pat_id), study_id)

def series_folder(
    pat_id: types.PatientID,
    study_id: str,
    modality: Literal['CT', 'RTDOSE', 'RTPLAN', 'RTSTRUCT']) -> str:
    return os.path.join(study_folder(pat_id, study_id), modality)

def image_filepath(
    pat_id: types.PatientID,
    study_id: str,
    modality: Literal['CT', 'RTDOSE', 'RTPLAN', 'RTSTRUCT'],
    sop_id: str) -> str:
    if modality == 'RTDOSE':
        prefix = 'RD'
    elif modality == 'RTPLAN':
        prefix = 'RP'
    elif modality == 'RTSTRUCT':
        prefix = 'RS'
    return os.path.join(series_folder(pat_id, study_id, modality), f'{prefix}.{sop_id}')

def raise_error(
    pat_id: str,
    study_id: str,
    modality: Literal['CT', 'RTDOSE', 'RTPLAN', 'RTSTRUCT'],
    sop_id: str) -> None:
    raise ValueError(f"No '{modality}' ({sop_id}) downloaded for patient '{pat_id}', study '{study_id}'.")
