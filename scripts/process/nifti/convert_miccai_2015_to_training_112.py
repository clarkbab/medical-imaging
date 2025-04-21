from mymi.processing.datasets.nifti import convert_to_segmentation_training_holdout

dataset = 'MICCAI-CROP'
dest_dataset = 'MICCAI-112'
kwargs = dict(
    spacing=(1, 1, 2)
)
convert_to_segmentation_training_holdout(dataset, dest_dataset, **kwargs)
