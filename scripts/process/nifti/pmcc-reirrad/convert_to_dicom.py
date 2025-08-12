from mymi.processing.nifti import convert_to_dicom

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    convert_dose=False,
    convert_landmarks=False,
    convert_regions=False,
)
convert_to_dicom(dataset, **kwargs)
