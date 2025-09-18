import numpy as np
import os
from tqdm import tqdm

from mymi import config
from mymi.datasets import NiftiDataset
from mymi.transforms import resample
from mymi.typing import *

def convert_to_training_unigradicon(
    dataset: DatasetID,
    splits: Splits = 'all') -> None:

    # pad_size = (355, 275, 329)
    pad_val = -2000
    ugi_size = (175, 175, 175)

    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(splits=splits)
    images = []
    spacings = []   # Required for (mm-based, sitk) data augmentation.
    for p in tqdm(pat_ids):
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        moving_study = pat.study('study_0')
        
        # Process moving CT.
        moving_ct = moving_study.ct_data
        output_spacing = np.array(moving_ct.shape) / ugi_size
        moving_ct_p = resample(moving_ct, output_size=ugi_size, output_spacing=output_spacing)
        # # UGI uses 'itk.imread' follwed by 'np.array(img)' to load training/inference data.
        # # 'itk.imread' expects '.nii' data to be in RAS+ coordinates and so sets negative 
        # # x/y directions/origins, whilst leaving image data unchanged. AFAIK, UGI doesn't use
        # # directions/origins, so it's presenting data in LPS+ coords. However 'np.array(img)'
        # # transposes axes, so the network actually sees SPL+ data. 
        # # Convert our fine-tuning data to match this format.

        # UGI is trained using heavy data augmentation that flips images on all axes, so it's
        # fine to present our data to the network in any orientation.
        # moving_ct_p = moving_ct_p.transpose()
        images.append(moving_ct_p)
        spacings.append(output_spacing)

        # Process fixed CT.
        fixed_ct = fixed_study.ct_data
        output_spacing = np.array(fixed_ct.shape) / ugi_size
        fixed_ct_p = resample(fixed_ct, output_size=ugi_size, output_spacing=output_spacing)
        # fixed_ct_p = fixed_ct_p.transpose()
        images.append(fixed_ct_p)
        spacings.append(output_spacing)

    # Save images as torch file.
    images = [torch.tensor(i)[None, None] for i in images]
    filepath = os.path.join(config.directories.datasets, 'training', dataset, 'data', 'unigradicon', 'train-samples.pt')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save([images, spacings], filepath)
