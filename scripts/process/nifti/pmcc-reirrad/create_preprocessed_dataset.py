from mymi.processing.datasets.nifti import create_corrfield_preprocessed_dataset

dataset = 'PMCC-REIRRAD'
new_dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    margin=20,
    # pat_ids='pat_0',
)

create_corrfield_preprocessed_dataset(dataset, new_dataset, **kwargs)
