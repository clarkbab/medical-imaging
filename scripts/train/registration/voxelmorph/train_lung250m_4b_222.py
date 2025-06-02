from mymi.training import train_voxelmorph

dataset = 'LUNG250M-4B-222'
model = 'voxelmorph-static'

train_voxelmorph(dataset, model)
