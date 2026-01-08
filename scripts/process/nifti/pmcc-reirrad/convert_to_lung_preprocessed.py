from mymi.processing.nifti import convert_to_lung_preprocessed_dataset

dataset = 'PMCC-REIRRAD'
dest_dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    margin_mm=20,
    recreate_patient=True,
)

convert_to_lung_preprocessed_dataset(dataset, dest_dataset, **kwargs)
