from augmed import Transform
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *

from mymi import datasets as ds
from dicomset.training import TrainingDataset
from mymi import logging
from mymi.typing import *
from mymi.utils.io import load_json, load_nifti, load_numpy, save_json
from mymi.utils.python import has_private_attr
from augmed.utils import save_json, to_tensor

from .random_sampler import RandomSampler


def create_training_projections(
    dataset: TrainingDataset,
    pat: PatientID,
    projection_geometry: Dict[str, Any],
    n_angles: int = 3,
    min_angle: float = 0,
    max_angle: float = 360,
    transform: Transform | None = None,
    epoch: int | None = None,
):
    trainpath = os.path.join(dataset.path, 'data', 'training', pat)

    # Save projection geometry for reproducibility.
    filepath = os.path.join(trainpath, 'log', 'geometry.json')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_json(projection_geometry, filepath)

    # Load volumes.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inh_ct, inh_affine = load_nifti(os.path.join(trainpath, 'volumes', 'inh_ct.nii.gz'))
    exh_ct, exh_affine = load_nifti(os.path.join(trainpath, 'volumes', 'exh_ct.nii.gz'))
    inh_labels = load_numpy(os.path.join(trainpath, 'volumes', 'inh_labels.npy'))
    exh_labels = load_numpy(os.path.join(trainpath, 'volumes', 'exh_labels.npy'))

    inh_ct = to_tensor(inh_ct, device=device)
    exh_ct = to_tensor(exh_ct, device=device)
    inh_affine = to_tensor(inh_affine, device=device)
    exh_affine = to_tensor(exh_affine, device=device)
    inh_labels = to_tensor(inh_labels, device=device)
    exh_labels = to_tensor(exh_labels, device=device)

    assert torch.all(inh_affine == exh_affine), "Inhale/exhale affines do not match."
    affine = inh_affine

    # Augment the volumes.
    if transform is not None:
        inh_ct, exh_ct, inh_labels, exh_labels, affine, params = transform(
            inh_ct, exh_ct, inh_labels, exh_labels,
            affine=affine, return_affine=True, return_params=True,
        )
        assert torch.all(affine == inh_affine), (
            "Affine was modified by the transform – spatial transforms should not change the affine."
        )
        if epoch is not None:
            filepath = os.path.join(trainpath, 'log', f'aug_params_{epoch}.json')
            save_json(params, filepath)

    # Sample kV source angles.
    kv_source_angles = list(np.random.uniform(min_angle, max_angle, n_angles))
    filepath = os.path.join(trainpath, 'log', f'kv_source_angles_{epoch}.json')
    save_json(kv_source_angles, filepath)

    # Generate projections.
    inh_ct_proj, inh_labels_proj = create_ctorch_projections(
        inh_ct, affine,
        projection_geometry['isocentre'],
        projection_geometry['sid'],
        projection_geometry['sdd'],
        projection_geometry['det_size'],
        projection_geometry['det_spacing'],
        projection_geometry['det_offset'],
        kv_source_angles,
        labels=inh_labels,
    )
    exh_ct_proj, exh_labels_proj = create_ctorch_projections(
        exh_ct, affine,
        projection_geometry['isocentre'],
        projection_geometry['sid'],
        projection_geometry['sdd'],
        projection_geometry['det_size'],
        projection_geometry['det_spacing'],
        projection_geometry['det_offset'],
        kv_source_angles,
        labels=exh_labels,
    )

    # Move projections to CPU numpy arrays for storage.
    inh_ct_proj = inh_ct_proj.cpu().numpy() if isinstance(inh_ct_proj, torch.Tensor) else np.asarray(inh_ct_proj)
    exh_ct_proj = exh_ct_proj.cpu().numpy() if isinstance(exh_ct_proj, torch.Tensor) else np.asarray(exh_ct_proj)
    inh_labels_proj = inh_labels_proj.cpu().numpy() if isinstance(inh_labels_proj, torch.Tensor) else np.asarray(inh_labels_proj)
    exh_labels_proj = exh_labels_proj.cpu().numpy() if isinstance(exh_labels_proj, torch.Tensor) else np.asarray(exh_labels_proj)

    return inh_ct_proj, exh_ct_proj, inh_labels_proj, exh_labels_proj, kv_source_angles

class DRRLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        pat: PatientID,
        batch_size: int = 32,
        n_train_angles: int = 3,
        n_val_angles: int = 3,
        n_val_volumes: int = 3,
        num_workers: int = 4,
        projection_geometry: Dict[str, Any] | None = None,
        transform_train: Transform | None = None,
        **kwargs,
        ) -> Tuple[DataLoader, DataLoader]:
        set = TrainingDataset(dataset)

        # Create train loader.
        train_set = TrainingSet(
            set, 
            pat,
            n_angles=n_train_angles,
            projection_geometry=projection_geometry,
            transform=transform_train,
        )
        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=num_workers, shuffle=True)

        # Create val loader.
        val_set = ValidationDataset(
            set,
            pat,
            n_angles=n_val_angles,
            n_volumes=n_val_volumes,
        )
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=num_workers, shuffle=False)

        return train_loader, val_loader

class TrainingSet(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        pat: PatientID,
        n_angles: int = 3,
        projection_geometry: Dict[str, Any] | None = None,
        standardise: bool = True,
        transform: Transform | None = None,
        **kwargs,
        ) -> None:
        self.__dataset = dataset
        self.__pat = pat
        self.__n_angles = n_angles
        self.__projection_geometry = projection_geometry
        self.__standardise = standardise
        self.__transform = transform

    def create_projections(self, epoch: int | None = None) -> None:
        """Generate projections and store them as in-memory attributes."""
        inh_ct_proj, exh_ct_proj, inh_labels_proj, exh_labels_proj, kv_source_angles = create_training_projections(
            dataset=self.__dataset,
            pat=self.__pat,
            projection_geometry=self.__projection_geometry,
            n_angles=self.__n_angles,
            transform=self.__transform,
            epoch=epoch,
        )
        self.__inh_ct_proj = inh_ct_proj
        self.__exh_ct_proj = exh_ct_proj
        self.__inh_labels_proj = inh_labels_proj
        self.__exh_labels_proj = exh_labels_proj
        self.__kv_source_angles = kv_source_angles

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[ProjImage, ProjLabelImage, Angle]:
        if not has_private_attr(self, '__inh_ct_proj'):
            raise ValueError("Projections have not been created. Call create_projections(epoch) at the start of the epoch.")

        # For n_angles=10, we have 20 samples due to inhale/exhale.
        if idx < self.__n_angles:
            angle_idx = idx
            data = self.__inh_ct_proj[angle_idx]
            labels = self.__inh_labels_proj[angle_idx]
        else:
            angle_idx = idx - self.__n_angles
            data = self.__exh_ct_proj[angle_idx]
            labels = self.__exh_labels_proj[angle_idx]

        angle = self.__kv_source_angles[angle_idx]

        # Convert to tensors.
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

        if self.__standardise:
            data = (data - torch.mean(data)) / torch.std(data)

        # Add singleton channel dimension.
        data = data[None, ...]

        # Add background class.
        label_bkg = 1 - torch.clamp(labels.sum(axis=0), 0, 1).unsqueeze(0)
        labels = torch.cat([label_bkg, labels], axis=0)

        return data, labels, angle

    def __len__(self):
        return self.__n_angles * 2  # Inhale/exhale.

class ValidationDataset(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        pat: PatientID,
        n_angles: int = 3,
        n_volumes: int = 3,
        standardise: bool = True,
        **kwargs,
        ) -> None:
        self.__n_angles = n_angles
        self.__n_volumes = n_volumes
        self.__standardise = standardise

        # Keep inhale/exhale volumes in memory.
        inh_ct_projs = []
        exh_ct_projs = []
        inh_labels_projs = []
        exh_labels_projs = []
        angleses = []
        for i in range(n_volumes):
            filepath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections', f"inh_ct_{i}.npy")
            inh_ct_proj = load_numpy(filepath)
            inh_ct_projs.append(inh_ct_proj)
            filepath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections', f"exh_ct_{i}.npy")
            exh_ct_proj = load_numpy(filepath)
            exh_ct_projs.append(exh_ct_proj)
            filepath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections', f"inh_labels_{i}.npy")
            inh_labels_proj = load_numpy(filepath)
            inh_labels_projs.append(inh_labels_proj)
            filepath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections', f"exh_labels_{i}.npy")
            exh_labels_proj = load_numpy(filepath)
            exh_labels_projs.append(exh_labels_proj)
            filepath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections', f"angles_{i}.json")
            kv_source_angles = load_json(filepath)
            angleses.append(kv_source_angles)
        self.__inh_ct_proj = np.stack(inh_ct_projs, axis=0)
        self.__exh_ct_proj = np.stack(exh_ct_projs, axis=0)
        self.__inh_labels_proj = np.stack(inh_labels_projs, axis=0)
        self.__exh_labels_proj = np.stack(exh_labels_projs, axis=0)
        self.__kv_source_angles = np.stack(angleses, axis=0)

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[ProjImage, ProjLabelImage, Angle]:
        if not has_private_attr(self, '__inh_ct_proj') or not has_private_attr(self, '__exh_ct_proj'):
            raise ValueError("Projections have not been created. Call create_projections(epoch) at the start of the epoch.")

        # For n_volumes=10, n_angles=10, we have 200 samples due to inhale/exhale.
        if idx < self.__n_volumes * self.__n_angles:
            # Get inhale image/labels.
            volume_idx = idx // self.__n_angles
            angle_idx = idx % self.__n_angles
            data = self.__inh_ct_proj[volume_idx, angle_idx]
            labels = self.__inh_labels_proj[volume_idx, angle_idx]
            angle = self.__kv_source_angles[volume_idx, angle_idx]
        else:
            # Get exhale image/labels.
            idx = idx - self.__n_volumes * self.__n_angles
            volume_idx = idx // self.__n_angles
            angle_idx = idx % self.__n_angles
            data = self.__exh_ct_proj[volume_idx, angle_idx]
            labels = self.__exh_labels_proj[volume_idx, angle_idx]
            angle = self.__kv_source_angles[volume_idx, angle_idx]

        if self.__standardise:
            data = (data - np.mean(data)) / np.std(data)

        # Add singleton channel dimension.
        data = data[None, ...]

        # Add background class.
        label_bkg = 1 - np.expand_dims(labels.sum(axis=0).clip(0, 1), axis=0)
        labels = np.concatenate([label_bkg, labels], axis=0)

        return data, labels, angle

    def __len__(self):
        return self.__n_volumes * self.__n_angles * 2  # Inhale/exhale.
