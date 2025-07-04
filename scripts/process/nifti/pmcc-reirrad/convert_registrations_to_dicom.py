from mymi.processing.nifti import convert_registration_predictions_to_dicom

dataset = 'PMCC-REIRRAD-CP'
model = 'deeds'
convert_registration_predictions_to_dicom(dataset, model)

