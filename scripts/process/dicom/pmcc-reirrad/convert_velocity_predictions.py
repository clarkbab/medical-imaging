from mymi.processing.dicom import convert_velocity_predictions_to_nifti

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    pat_ids=['pat_5', 'pat_6'],
)
convert_velocity_predictions_to_nifti(dataset, **kwargs)
