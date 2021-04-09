import os

root_dir = os.environ['MYMI_DATA']

# Cache.
cache_dir = os.path.join(root_dir, 'cache')

# Checkpoints.
checkpoint_dir = os.path.join(root_dir, 'checkpoints')

# Datasets.
dataset_dir = os.path.join(root_dir, 'datasets')

# Figures.
figure_dir = os.path.join(root_dir, 'figures')

# Tensorboard.
tensorboard_dir = os.path.join(root_dir, 'tensorboard')
