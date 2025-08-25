from mymi.processing.nifti import convert_to_training_unigradicon

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    splits='train',
)
convert_to_training_unigradicon(dataset, **kwargs)
