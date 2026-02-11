from mymi.datasets.dicom import DicomStudy
from mymi.processing.dicom import convert_to_nifti

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
    dry_run=False,
    # group='hn',
    pat=[
        'PMCC_ReIrrad_L14',
    ],
    recreate=False,
    recreate_ct=True,
    recreate_dose=True,
    recreate_patient=False,
    recreate_landmarks=True,
    # region='rl:pmcc-reirrad-lung',
    study_sort=study_sort,
)
convert_to_nifti(dataset, **kwargs)
