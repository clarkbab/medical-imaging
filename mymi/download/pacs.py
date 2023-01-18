from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
from pydicom.dataset import Dataset
from pynetdicom import AE
from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
from typing import List, Literal, Tuple

from mymi import config
from mymi import logging
from mymi import types

# To download images from PACS, your desktop must have a static IP configured
# and a record should be created in PACS admin to allow your images to be downloaded
# to this IP.
PACS_AE_TITLE = 'DCMTK_A40942'      # 'AE' (application entity) field in PACS admin.
PACS_PORT = 11112                   # Incoming port - on our desktop.
PACS_REMOTE_AE_TITLE = 'pacsFIR'    # AE (application) entity title of remote PACS server.
PACS_REMOTE_IP = '10.126.42.16'
PACS_REMOTE_PORT = 2104 

def download_dicoms(
    filepath: str,
    date_format: str = '%d/%m/%Y',
    sep: str = '\t') -> None:
    # Load file.
    df = pd.read_csv(filepath, sep=sep)

    # Download files for each patient/date.
    for _, (pat_id, ct_date) in df.iterrows():
        ct_date = datetime.strptime(ct_date, date_format)
        download_patient_dicoms(pat_id, ct_date)

def download_patient_dicoms(
    pat_id: types.PatientID,
    ct_date: datetime) -> None:
    download_patient_rtdose_dicoms(pat_id, ct_date)

def download_patient_rtdose_dicoms(
    pat_id: types.PatientID,
    ct_date: datetime) -> None:
    logging.arg_log("Downloading RTDOSE dicoms", ('pat_id', 'ct_date'), (pat_id, ct_date))

    # Search for RTDOSE files by date.
    query_results = queryRTSeries(pat_id, ct_date, 'RTDOSE')

    for study_id, series_id in query_results:
        # Create dose folder.
        folder = dose_folder(pat_id, study_id)
        os.makedirs(folder, exist_ok=True)

        # Download RTDOSE file.
        command = f"""
movescu --study --key QueryRetrieveLevel=SERIES --key PatientID={pat_id} --key StudyInstanceUID='{study_id}' \
--key SeriesInstanceUID='{series_id}' --aetitle {PACS_AE_TITLE} --call {PACS_REMOTE_AE_TITLE} {PACS_REMOTE_IP} \
{PACS_REMOTE_PORT} --move {PACS_AE_TITLE} --output-directory {folder} --port {PACS_PORT}
        """
        logging.info(command)
        os.system(command)

def queryRTSeries(
    pat_id: types.PatientID,
    ct_date: datetime,
    modality: Literal['RTDOSE', 'RTPLAN', 'RTSTRUCT']) -> List[Tuple[str, str]]:
    # Initialise the Application Entity
    ae = AE()

    # Add a requested presentation context
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)

    # Create our Identifier (query) dataset
    ds = Dataset()
    ds.PatientID = str(pat_id)
    ds.QueryRetrieveLevel = 'IMAGE'
    ds.Modality = modality
    ds.StudyDate = '*'
    ds.SeriesInstanceUID = '*'

    results = []

    time_delta=relativedelta(days=0)

    # Associate with the peer at IP address and port
    assoc = ae.associate(PACS_REMOTE_IP, PACS_REMOTE_PORT, ae_title=PACS_REMOTE_AE_TITLE)

    if assoc.is_established:
        # Use the C-FIND service to send the identifier
        responses = assoc.send_c_find(ds, StudyRootQueryRetrieveInformationModelFind)

        for (status, identifier) in responses:
            print(identifier)
            if status:
                #print('C-FIND query status: 0x{0:04x}'.format(status.Status))

                # If the status is 'Pending' then identifier is the C-FIND response
                if status.Status in (0xFF00, 0xFF01):
                    # Keep series that are within the allowed date range.
                    date = parse_date(identifier.StudyDate)
                    if date >= ct_date - time_delta and date <= ct_date:
                        results.append([identifier.StudyInstanceUID, identifier.SeriesInstanceUID])
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

def dose_folder(
    pat_id: types.PatientID,
    study_id: str):
    return os.path.join(study_folder(pat_id, study_id), 'RTDOSE')
