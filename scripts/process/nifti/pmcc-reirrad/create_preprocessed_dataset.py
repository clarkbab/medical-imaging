from mymi.processing.datasets.nifti import create_preprocessed_lung_dataset

dataset = 'PMCC-REIRRAD'
new_dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    margin=20,
    # pat_ids='pat_0',
)

create_preprocessed_lung_dataset(dataset, new_dataset, **kwargs)
