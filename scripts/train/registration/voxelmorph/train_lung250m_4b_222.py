from mymi.training import train_voxelmorph

dataset = 'LUNG250M-4B-222'
model = 'dynamic-2000'
kwargs = dict(
    pad_shape=(173, 132, 226),
)

train_voxelmorph(dataset, model, **kwargs)
