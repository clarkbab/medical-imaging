from mymi.processing.nifti import create_lung_preprocessed_dataset

dataset = 'PMCC-REIRRAD'
dest_dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    margin_mm=20,
    # pat_ids='pat_0',
)

create_lung_preprocessed_dataset(dataset, dest_dataset, **kwargs)
