from mymi.processing.nifti import convert_to_unigradicon_training

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    splits='train',
)
convert_to_unigradicon_training(dataset, **kwargs)
