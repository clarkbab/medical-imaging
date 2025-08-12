from mymi.processing.nifti import convert_registration_predictions_to_dicom

dataset = 'PMCC-REIRRAD'
model = 'deeds'
kwargs = dict(
    convert_dose=True,
    convert_moved=True,
    landmark_ids=None,
    pat_ids='PMCC_ReIrrad_L01',
    region_ids=None,
)
convert_registration_predictions_to_dicom(dataset, model, **kwargs)
