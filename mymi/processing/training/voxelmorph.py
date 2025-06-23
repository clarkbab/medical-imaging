import numpy as np
import os
from tqdm import tqdm

from mymi.datasets import TrainingDataset
from mymi.transforms import centre_pad
from mymi import logging

def create_voxelmorph_training_data(dataset: str) -> None:
    # Load sample IDs.
    set = TrainingDataset(dataset)
    splits = ['train', 'validate']
    sample_ids = set.list_samples(splits=splits)

    # We altered VXM code so that input size doesn't need to be consistent.
    # # Get largest image.
    # size = [0, 0, 0]
    # for s in sample_ids:
    #     input_shape = set.sample(s).input.shape[1:]
    #     for i, ss in enumerate(input_shape):
    #         if ss > size[i]:
    #             size[i] = ss
    # size = tuple(size)
    # logging.info(f"Padding all images to size '{size}'.")

    # Write data.
    vxm_datapath = os.path.join(set.path, 'vxm-data')
    os.makedirs(vxm_datapath, exist_ok=True)
    index_paths = []
    image_idx = 0
    for s in tqdm(sample_ids):
        sample = set.sample(s)
        input = sample.input
        # input = centre_pad(input, size)
        fixed_path = os.path.join(vxm_datapath, f'{image_idx}.npz')
        np.savez_compressed(fixed_path, vol=input[0])
        index_paths.append(fixed_path)
        image_idx += 1
        moving_path = os.path.join(vxm_datapath, f'{image_idx}.npz')
        np.savez_compressed(moving_path, vol=input[1])
        index_paths.append(moving_path)
        image_idx += 1

    # Write index.
    filepath = os.path.join(set.path, 'vxm-index.txt')
    with open(filepath, 'w') as f:
        for p in index_paths:
            f.write(f"{p}\n")
