from dicomset.dicom import DicomStudy
from dicomset.dicom.utils import convert_to_nifti

def study_sort(s: DicomStudy) -> int:
    if not s.has_rtstruct:
        raise ValueError(f"No RTSTRUCT series found for study '{s}'.")
    filepath = s.rtstruct_series('i:0').filepath
    if '/C1/' in filepath:
        return 0
    elif '/C2/' in filepath or '/C2_PROP/':
        return 1
    raise ValueError(f"Study is not C1 or C2. Filepath: {filepath}")

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    anonymise_patients=False,
    # group='hn',
    recreate_ct=False,
    recreate_dataset=False,
    recreate_dose=False,
    recreate_landmarks=True,
    recreate_regions=False,
    recreate_patients=False,
    # recreate_patient_id='PMCC_ReIrrad_H17:',
    region_id='l:hn+lung',
    sort_studies=study_sort,
)
convert_to_nifti(dataset, **kwargs)
