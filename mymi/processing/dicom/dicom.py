import matplotlib as mpl
import os
import shutil
from tqdm import tqdm
from typing import *

from mymi.datasets import DicomDataset
from mymi.datasets.dicom import ROIData, RtStructConverter, recreate as recreate_dicom
from mymi import logging
from mymi.typing import *
from mymi.utils import *

from .ct import to_ct_dicoms
from .rtdose import to_rtdose_dicom
from .rtstruct import to_rtstruct_dicom

def convert_to_dicom(
    set: 'Dataset',
    convert_ct: bool = True,
    convert_dose: bool = True,
    convert_landmarks: bool = True,
    convert_regions: bool = True,
    dest_dataset: Optional[str] = None,
    landmark_ids: LandmarkIDs = 'all',
    landmarks_prefix: Optional[str] = 'Marker',
    pat_ids: PatientIDs = 'all',
    pat_id_map: Optional[Dict[PatientID, PatientID]] = None,
    pat_prefix: Optional[str] = None,
    recreate_dataset: bool = False,
    region_ids: RegionIDs = 'all') -> None:

    # Create destination folder.
    dest_dataset = set.id if dest_dataset is None else dest_dataset
    if recreate_dataset:
        destset = recreate_dicom(dest_dataset)
    else:
        destset = DicomDataset(dest_dataset)

    # Remove old patients data.
    base_path = os.path.join(destset.path, 'data', 'patients')
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # Load patients.
    pat_ids = set.list_patients(pat_ids=pat_ids)

    for p in tqdm(pat_ids):
        logging.info(p)
        # Map patient ID and apply prefix.
        pat = set.patient(p)
        p_mapped = pat_id_map[p] if pat_id_map is not None and p in pat_id_map else p
        p_mapped = f'{pat_prefix}{p_mapped}' if pat_prefix is not None else p_mapped

        study_ids = pat.list_studies()
        for s in study_ids:
            logging.info(s)
            study = pat.study(s)

            # Need to store these so that RTSTRUCT file can reference CT DICOM files.
            # Means we can only process studies that have a single CT series - as the code stands.
            ct_dicoms = None

            # Convert CT series.
            ct_series_ids = study.list_series('ct')
            if len(ct_series_ids) > 1:
                raise ValueError(f"Code only handles studies with a single CT series. See 'ct_dicoms' above.")
            ct_series_id = ct_series_ids[0]
            # Load data.
            ct_series = study.ct(ct_series_id)
            ct_dicoms = to_ct_dicoms(ct_series.data, ct_series.spacing, ct_series.offset, p_mapped, s)
            for i, d in enumerate(ct_dicoms):
                if convert_ct:
                    filepath = os.path.join(base_path, p_mapped, s, 'ct', ct_series_id, f'{i:03d}.dcm')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    d.save_as(filepath)

            # Convert regions to RTSTRUCT.
            if (convert_landmarks and study.has_landmarks_series) or (convert_regions and study.has_regions_series):
                all_series_ids = list(np.unique(study.list_series('landmarks') + study.list_series('regions')))
                for ss in all_series_ids:
                    lm_data = study.series(ss, 'landmarks').data() if convert_landmarks and study.has_series(ss, 'landmarks') else None
                    r_data = study.series(ss, 'regions').data() if convert_regions and study.has_series(ss, 'regions') else None
                    rtstruct_dicom = to_rtstruct_dicom(ct_dicoms, landmarks_data=lm_data, regions_data=r_data)
                    filepath = os.path.join(base_path, p_mapped, s, 'rtstruct', f'{ss}.dcm')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    if not os.path.exists(filepath):
                        rtstruct_dicom.save_as(filepath)

            # # Convert landmarks (voxel coordinates) to fCSV (used by Slicer).
            # if convert_landmarks and pat.has_landmarks:
            #     lms_voxel = pat.landmarks
                
            #     # Convert to patient coordinates.
            #     filepath = os.path.join(destset.path, 'data', p_mapped, s, 'landmarks.fcsv') 
            #     os.makedirs(os.path.dirname(filepath), exist_ok=True)
            #     with open(filepath, 'w') as f:
            #         slicer_version = '5.0.2'
            #         f.write(f'# Markups fiducial file version = {slicer_version}\n')
            #         f.write(f'# CoordinateSystem = LPS\n')
            #         f.write(f'# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n')

            #         for i, lm_voxel in enumerate(lms_voxel):
            #             lm_patient = np.array(lm_voxel) * spacing + offset
            #             f.write(f'{i},{lm_patient[0]},{lm_patient[1]},{lm_patient[2]},0,0,0,1,1,1,0,{i},,\n')

            # Convert dose to RTDOSE/RTPLAN.
            if convert_dose and study.has_dose:
                dose_series_ids = study.list_series('dose')
                for ss in dose_series_ids:
                    dose_series = study.series(ss, 'dose')
                    dose_dicom = to_rtdose_dicom(dose_series.data, dose_series.spacing, dose_series.offset, ct_dicoms[0])
                    filepath = os.path.join(base_path, p_mapped, s, 'rtdose', f'{ss}.dcm')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    dose_dicom.save_as(filepath)
