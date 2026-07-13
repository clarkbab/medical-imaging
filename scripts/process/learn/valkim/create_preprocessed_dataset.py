from mymi.processing.nifti.valkim import create_valkim_preprocessed_dataset

kwargs = dict(
    blur_markers=False,
    patient_id='PAT3',
    recreate_dataset=False,
)
create_valkim_preprocessed_dataset(**kwargs)
