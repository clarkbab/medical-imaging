from mymi.processing.nifti.valkim import create_valkim_preprocessed_dataset

kwargs = dict(
    patient_id='PAT1',
    recreate=False,
)
create_valkim_preprocessed_dataset(**kwargs)
