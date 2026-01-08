import os
from tqdm import tqdm

from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate as recreate_nifti
from mymi.utils import *

def convert_popi_to_nifti(dry_run: bool = False) -> None:
    dataset = 'POPI'
    fixed_study = 'study_1'
    moving_study = 'study_0'
    exhale_phases = [50, 60, 50, 50, 50, 50]
    inhale_phase = 0
    landmarks_y_origin = 0

    set = RawDataset(dataset)
    if not dry_run:
        dest_set = recreate_nifti(dataset)
    basepath = os.path.join(set.path, 'MedPhys11')
    pat_ids = list(sorted(os.listdir(basepath)))
    for i, p in tqdm(enumerate(pat_ids), total=len(pat_ids)):
        # Save fixed (inhale) image.
        filepath = os.path.join(basepath, p, 'mhd', f'{inhale_phase:02}.mhd')
        data, spacing, origin = sitk_load_image(filepath)
        if not dry_run:
            filepath = os.path.join(dest_set.path, 'data', 'patients', p, fixed_study, 'ct', 'series_0.nii.gz')
            save_nifti(data, filepath, spacing=spacing, origin=origin)
        
        # Save moving (exhale) image.
        exhale_phase = exhale_phases[i]
        filepath = os.path.join(basepath, p, 'mhd', f'{exhale_phase:02}.mhd')
        data, spacing, origin = sitk_load_image(filepath)
        if not dry_run:
            filepath = os.path.join(dest_set.path, 'data', 'patients', p, moving_study, 'ct', 'series_0.nii.gz')
            save_nifti(data, filepath, spacing=spacing, origin=origin)
            
        # Save fixed (inhale) landmarks.
        filepath = os.path.join(basepath, p, 'pts', f'{inhale_phase:02}.pts')
        landmarks = pd.read_csv(filepath, sep='[\s\t]', header=None, engine='python')
        landmarks[1] += landmarks_y_origin
        landmarks.insert(0, 'landmark-id', list(range(len(landmarks))))
        if not dry_run:
            filepath = os.path.join(dest_set.path, 'data', 'patients', p, fixed_study, 'landmarks', 'series_1.csv')
            save_csv(landmarks, filepath)
            
        # Save moving (exhale) landmarks.
        filepath = os.path.join(basepath, p, 'pts', f'{exhale_phase:02}.pts')
        landmarks = pd.read_csv(filepath, sep='[\s\t]', header=None, engine='python')
        landmarks[1] += landmarks_y_origin
        landmarks.insert(0, 'landmark-id', list(range(len(landmarks))))
        if not dry_run:
            filepath = os.path.join(dest_set.path, 'data', 'patients', p, moving_study, 'landmarks', 'series_1.csv')
            save_csv(landmarks, filepath)
