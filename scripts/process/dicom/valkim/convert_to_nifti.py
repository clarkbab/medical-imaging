from mymi.datasets.dicom import DicomStudy
from mymi.processing.dicom import convert_to_nifti

dataset = 'VALKIM'
pat_ids = [
    'PAT1',
    'PAT2',
    'PAT3',
]
regions = [
    'Fiducial_1',
    'Fiducial_2',
    'Fiducial_3',
    'GTV_Exh',
    'GTV_Inh',
]
kwargs = dict(
    pat=pat_ids,
    makeitso=True,
    recreate=True,
    recreate_ct=True,
    recreate_dose=True,
    recreate_patient=True,
    recreate_landmarks=True,
    region=regions,
)
convert_to_nifti(dataset, **kwargs)
