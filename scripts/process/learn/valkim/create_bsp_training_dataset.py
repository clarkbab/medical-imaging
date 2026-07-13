from dicomset.utils import save_numpy
from dicomset.training.utils import create_dataset
from dicomset.typing import *
import numpy as np
import os
from tqdm.auto import tqdm

from mymi.utils.cdog import load_patient, load_patient_shrouds

breathing_signals = ['MMAmplitude0', 'MMPhase0']

def create_bsp_training_dataset(
    split_seed: int = 42,
    ) -> None:
    dataset = 'VALKIM-BSP'
    pat_ids = ['Patient01', 'Patient02']

    # Create training dataset.
    set = create_dataset(dataset)

    datapath = os.path.join(set.path, 'data')
    trainpath = os.path.join(datapath, 'train')
    valpath = os.path.join(datapath, 'validation')
    testpath = os.path.join(datapath, 'test')
    os.makedirs(trainpath, exist_ok=True)
    os.makedirs(valpath, exist_ok=True)
    os.makedirs(testpath, exist_ok=True)

    # Seed RNG for reproducible splits.
    rng = np.random.default_rng(split_seed)
    train = 0.6
    val = 0.2
    test = 0.2
    print(f'Using random seed {split_seed} for dataset splits with proportions: train={train}, val={val}, test={test}.')

    train_i = 0
    val_i = 0
    test_i = 0
    for p in tqdm(pat_ids, desc='Patients'):
        # Load patient shrouds.
        dirpath = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\{p}"
        data, _ = load_patient_shrouds(dirpath, load_data=True)
        n_fractions = len(data)

        # Load patient breathing signals.
        _, pat_info = load_patient(dirpath, load_data=False)
        points = []
        for i in pat_info:
            ps = []
            for j in i:
                s = j[breathing_signals].values
                
                # Convert to points in image space and normalise y-coordinates.
                p = signals_to_points(s)
                p = normalise_points(p)
                ps.append(p)
            points.append(ps)

        # Save shrouds.
        for i in range(n_fractions):
            frac_data = data[i]
            frac_points = points[i]
            n_arcs = len(frac_data)
            for j in range(n_arcs):
                arc_data = frac_data[j]
                arc_points = frac_points[j]

                # Assign to train/val/test split.
                draw = rng.random()
                if draw < train:
                    split = 'train'
                    sample_i = train_i
                    train_i += 1 
                elif draw < train + val:
                    split = 'validation'
                    sample_i = val_i
                    val_i += 1
                else:
                    split = 'test'
                    sample_i = test_i
                    test_i += 1
                destpath = os.path.join(datapath, split, f"{sample_i:03d}.npz")
                save_numpy([arc_data, arc_points], destpath, keys=['shroud', 'signals'], overwrite=True)

# Convert signals to points in image space so that they can be augmented with the images.
def normalise_points(
    points: BatchPoints2D,
    min: float = 0,
    max: float = 1,
    ) -> BatchPoints2D:
    
    norm_points = []
    for p in points:
        # y-axis only.
        y_min, y_max = p[:, 1].min(), p[:, 1].max()
        p[:, 1] = (max - min) * (p[:, 1] - y_min) / (y_max - y_min) + min
        norm_points.append(p)
    return np.stack(norm_points, axis=0)

# Adds the x-coordinates to 1D signals.
def signals_to_points(
    signals: np.ndarray,  # N x (number of signals).
    ) -> BatchPoints2D:
    n_signals = signals.shape[1]
    points = []
    for i in range(n_signals):
        signal = signals[:, i]
        p = np.stack([np.arange(len(signal)), signal], axis=1)
        points.append(p)
    return np.stack(points, axis=0)

if __name__ == '__main__':
    create_bsp_training_dataset()
