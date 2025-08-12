import numpy as np
import os
from tqdm import tqdm

from mymi import config
from mymi.datasets import NiftiDataset
from mymi.transforms import resample
from mymi.typing import *

def convert_to_unigradicon_training(
    dataset: DatasetID,
    splits: Splits = 'all') -> None:

    # pad_size = (355, 275, 329)
    pad_val = -2000
    ugi_size = (175, 175, 175)

    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(splits=splits)
    images = []
    for p in tqdm(pat_ids):
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        moving_study = pat.study('study_0')
        
        # Process moving CT.
        moving_ct = moving_study.ct_data
        output_spacing = np.array(moving_ct.shape) / ugi_size
        moving_ct_p = resample(moving_ct, output_size=ugi_size, output_spacing=output_spacing)
        images.append(moving_ct_p)

        # Process fixed CT.
        fixed_ct = fixed_study.ct_data
        output_spacing = np.array(fixed_ct.shape) / ugi_size
        fixed_ct_p = resample(fixed_ct, output_size=ugi_size, output_spacing=output_spacing)
        images.append(fixed_ct_p)

    # Save images as torch file.
    images = [torch.tensor(i)[None, None] for i in images]
    filepath = os.path.join(config.directories.datasets, 'training', dataset, 'data', 'unigradicon', 'train-samples.pt')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(images, filepath)
