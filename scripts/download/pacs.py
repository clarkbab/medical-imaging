from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
from typing import List, Literal, Optional, Tuple, Union

from mymi import config
from mymi import logging
from mymi import typing
from mymi.utils import append_row, load_files_csv, save_csv

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
    df = load_files_csv('patient-specific-models', 'data', 'urn.txt', names=['patient-id', 'study-date'], header=None, sep=sep)

    # Create error table.
    cols = {
        'patient-id': str,
        'study-id': str,
        'study-date': datetime,
        'modality': str,
        'series-id': str,
        'sop-id': str,
        'parent-modality': str,
        'parent-sop-id': str,
    }
    error_df = pd.DataFrame(columns=cols.keys())

    # Download files for each patient/date.
    for _, (pat_id, study_date) in df.iterrows():
        study_date = datetime.strptime(study_date, date_format)
        error_df = download_patient_dicoms(pat_id, study_date, error_df)

    # Save error file.
    save_csv(error_df, 'patient-specific-models', 'data', 'errors.csv', overwrite=True)

def download_patient_dicoms(
    pat_id: typing.PatientID,
    study_date: datetime,
    error_df: pd.DataFrame) -> None:
    logging.arg_log("Downloading patient DICOMS", ('pat_id', 'study_date'), (pat_id, study_date))

    # Download RTDOSE files. Allow 1 week for creation of these after CT creation.
    start_date = study_date
    end_date = study_date + relativedelta(days=RTDOSE_DELAY_DAYS)
    query_results = download_patient_rtdose_dicoms(pat_id, start_date, end_date)

    if len(query_results) == 0:
        # Add error entry.
        data = {
            'patient-id': pat_id,
            'study-date': study_date,
            'modality': 'rtdose'
        }
        error_df = append_row(error_df, data)
    
    # Download RTPLAN for each RTDOSE file.
    for study_id, _, rtdose_sop_id in query_results:
        # Get RTPLAN ID.
        filepath = image_filepath(pat_id, study_id, 'rtdose', rtdose_sop_id)
        rtdose = pydicom.read_file(filepath)
        rtplan_id = rtdose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID

        # Query for RTPLAN file.
        query_results = download_patient_rtplan_dicoms(pat_id, study_id, rtplan_id)
        
        if len(query_results) == 0:
            # Add error entry.
            data = {
                'patient-id': pat_id,
                'study-id': study_id,
                'study-date': study_date,
                'modality': 'rtplan',
                'sop-id': rtplan_id,
                'parent-modality': 'rtdose',
                'parent-sop-id': rtdose_sop_id
            }
            error_df = append_row(error_df, data)

        # Download RTSTRUCT for each RTPLAN file.
        for _, _, rtplan_sop_id in query_results:
            # Get RTSTRUCT ID.
            filepath = image_filepath(pat_id, study_id, 'rtplan', rtplan_sop_id)
            rtplan = pydicom.read_file(filepath)
            rtstruct_id = rtplan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID

            # Query for RTSTRUCT file.
            query_results = download_patient_rtstruct_dicoms(pat_id, study_id, rtstruct_id)

            if len(query_results) == 0:
                # Add error entry.
                data = {
                    'patient-id': pat_id,
                    'study-id': study_id,
                    'study-date': study_date,
                    'modality': 'rtstruct',
                    'sop-id': rtstruct_id,
                    'parent-modality': 'rtplan',
                    'parent-sop-id': rtplan_sop_id
                }
                error_df = append_row(error_df, data)

            # Download CT series for each RTSTRUCT file.
            for _, _, rtstruct_sop_id in query_results:
                # Get CT series ID.
                filepath = image_filepath(pat_id, study_id, 'rtstruct', rtstruct_sop_id)
                rtstruct = pydicom.read_file(filepath)
                ct_series_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID

                # Query CT files.
                query_results = download_patient_ct_dicoms(pat_id, study_id, ct_series_id)

                if len(query_results) == 0:
                    # Add error entry.
                    data = {
                        'patient-id': pat_id,
                        'study-id': study_id,
                        'study-date': study_date,
                        'modality': 'ct',
                        'series-id': ct_series_id,
                        'parent-modality': 'rtstruct',
                        'parent-sop-id': rtstruct_sop_id,
                    }
                    error_df = append_row(error_df, data)

    return error_df

def download_patient_rtdose_dicoms(
    pat_id: typing.PatientID,
    start_date: datetime,
    end_date: datetime) -> List[Tuple[str, str]]:
    logging.arg_log("Downloading patient RTDOSE dicoms", ('pat_id', 'start_date', 'end_date'), (pat_id, start_date, end_date))

    # Search for RTDOSE files by date.
    query_results = queryPACS(pat_id, 'rtdose', start_date=start_date, end_date=end_date)
    print(query_results)

    for study_id, series_id, sop_id in query_results:
        # Create RTDOSE folder.
        folder = series_folder(pat_id, study_id, 'rtdose')
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
        filepath = image_filepath(pat_id, study_id, 'rtdose', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'rtdose', sop_id)

    return query_results

def download_patient_rtplan_dicoms(
    pat_id: typing.PatientID,
    study_id: str,
    rtplan_id: str) -> None:
    logging.arg_log("Downloading patient RTPLAN dicoms", ('pat_id', 'study_id', 'rtplan_id'), (pat_id, study_id, rtplan_id))

    # Query for RTPLAN dicoms.
    query_results = queryPACS(pat_id, 'rtplan', sop_id=rtplan_id, study_id=study_id) 

    for study_id, series_id, sop_id in query_results:
        # Create RTPLAN folder.
        folder = series_folder(pat_id, study_id, 'rtplan')
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
        filepath = image_filepath(pat_id, study_id, 'rtplan', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'rtplan', sop_id)

    return query_results

def download_patient_rtstruct_dicoms(
    pat_id: typing.PatientID,
    study_id: str,
    rtstruct_id: str) -> None:
    logging.arg_log("Downloading patient RTSTRUCT dicoms", ('pat_id', 'study_id', 'rtstruct_id'), (pat_id, study_id, rtstruct_id))

    # Query for RTSTRUCT dicoms.
    query_results = queryPACS(pat_id, 'rtstruct', sop_id=rtstruct_id, study_id=study_id) 

    for study_id, series_id, sop_id in query_results:
        # Create RTSTRUCT folder.
        folder = series_folder(pat_id, study_id, 'rtstruct')
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
        filepath = image_filepath(pat_id, study_id, 'rtstruct', sop_id)
        if not os.path.exists(filepath):
            raise_error(pat_id, study_id, 'rtstruct', sop_id)

    return query_results

def download_patient_ct_dicoms(
    pat_id: typing.PatientID,
    study_id: str,
    ct_series_id: str) -> List[Tuple[str, str]]:
    logging.arg_log("Downloading patient CT dicoms", ('pat_id', 'study_id', 'ct_series_id'), (pat_id, study_id, ct_series_id))

    # Query for CT dicoms.
    query_results = queryPACS(pat_id, 'ct', series_id=ct_series_id, study_id=study_id)
    print(query_results)

    for study_id, series_id in query_results:
        # Create CT folder.
        folder = series_folder(pat_id, study_id, 'ct')
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
        folder = series_folder(pat_id, study_id, 'ct')
        files = os.listdir(folder)
        if len(files) == 0:
            raise_error(pat_id, study_id, 'ct', series_id)

    return query_results

def queryPACS(
    pat_id: typing.PatientID,
    modality: Literal['ct', 'rtdose', 'rtplan', 'rtstruct'],
    end_date: Optional[datetime] = None,
    series_id: str = '*',
    sop_id: str = '*',
    start_date: Optional[datetime] = None,
    study_id: str = '*') -> Union[List[Tuple[str, str]], List[Tuple[str, str]]]:
    # Determine query retrieve level.
    if modality == 'ct':
        level = 'SERIES'
    elif modality in ('rtdose', 'rtplan', 'rtstruct'):
        level = 'IMAGE'

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
        ds.StudyDate = '*'      # Filter dates in python - faster than using the query.
        ds.SeriesInstanceUID = series_id
        ds.StudyInstanceUID = study_id
        if level == 'IMAGE':
            ds.SOPInstanceUID = sop_id

        print('=== query ===')
        print(ds)

        # Use the C-FIND service to send the identifier
        responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)

        # Filter responses based on date.
        def response_filter(response):
            status, identifier = response
            # Filter on status.
            if status.Status not in (0xFF00, 0xFF01):
                return False

            # Filter on date.
            study_dt = datetime.strptime(identifier.StudyDate, PACS_DT_FORMAT)
            if start_date is not None and study_dt < start_date:
                return False
            elif end_date is not None and study_dt > end_date:
                return False
            return True
        responses = filter(response_filter, responses)

        for (status, identifier) in responses:
            if status:
                #print('C-FIND query status: 0x{0:04x}'.format(status.Status))

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
    pat_id: typing.PatientID,
    study_id: str):
    return os.path.join(config.directories.files, 'patient-specific-models', 'data', 'dcmFiles', str(pat_id), study_id)

def series_folder(
    pat_id: typing.PatientID,
    study_id: str,
    modality: Literal['ct', 'rtdose', 'rtplan', 'rtstruct']) -> str:
    return os.path.join(study_folder(pat_id, study_id), modality)

def image_filepath(
    pat_id: typing.PatientID,
    study_id: str,
    modality: Literal['ct', 'rtdose', 'rtplan', 'rtstruct'],
    sop_id: str) -> str:
    if modality == 'rtdose':
        prefix = 'RD'
    elif modality == 'rtplan':
        prefix = 'RP'
    elif modality == 'rtstruct':
        prefix = 'RS'
    return os.path.join(series_folder(pat_id, study_id, modality), f'{prefix}.{sop_id}')

def raise_error(
    pat_id: str,
    study_id: str,
    modality: Literal['ct', 'rtdose', 'rtplan', 'rtstruct'],
    sop_id: str) -> None:
    raise ValueError(f"No '{modality}' ({sop_id}) downloaded for patient '{pat_id}', study '{study_id}'.")
