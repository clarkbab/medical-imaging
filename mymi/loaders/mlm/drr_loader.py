from augmed import Transform
from dicomset.utils import logger
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from dicomset.typing import *
from dicomset.training import TrainingDataset
from dicomset.utils import load_json, load_nifti, load_numpy
from augmed.utils import save_json
from mymi.utils.python import has_private_attr
from mymi.utils.projections import project_ctorch
from augmed.utils import to_tensor

class DRRLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        pat: PatientID,
        batch_size: int = 32,
        # There are only 2 train volumes, these are augmented and then projected
        # at the start of each epoch.
        n_train_angles: int = 3,
        n_val_angles: int = 3,
        n_val_volumes: int = 3,
        num_workers: int = 0,
        preload_val_data: bool = True,
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
            preload=preload_val_data,
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
        preload: bool = False,
        **kwargs,
        ) -> None:
        self.__dataset = dataset
        self.__pat = pat
        self.__n_angles = n_angles
        self.__projection_geometry = projection_geometry
        self.__standardise = standardise
        self.__transform = transform
        self.__preload = preload

    def create_projections(self, epoch: int | None = None) -> None:
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
        ) -> Tuple[BatchImage2D, BatchLabelImage2D, List[float]]:
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
        preload: bool = False,
        **kwargs,
        ) -> None:
        self.__n_angles = n_angles
        self.__n_volumes = n_volumes
        self.__standardise = standardise
        self.__preload = preload

        dirpath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections')
        self.__inh_ct_files = sorted(
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.startswith('inh_ct_')
        )
        self.__inh_labels_files = sorted(
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.startswith('inh_labels_')
        )
        self.__exh_ct_files = sorted(
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.startswith('exh_ct_')
        )
        self.__exh_labels_files = sorted(
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.startswith('exh_labels_')
        )
        self.__angle_files = sorted(
            os.path.join(dirpath, f)
            for f in os.listdir(dirpath)
            if f.startswith('angles_')
        )

        if not self.__inh_ct_files:
            raise ValueError(f"No inhale CT projection files found in {dirpath}")

        self.__inh_ct_proj = None
        self.__inh_labels_proj = None
        self.__exh_ct_proj = None
        self.__exh_labels_proj = None
        self.__kv_source_angles = None

        if self.__preload:
            self._preload_data()

    def _preload_data(self) -> None:
        logger.info('Pre-loading validation data')
        n = self.__n_volumes
        self.__inh_ct_proj      = np.stack([load_numpy(f) for f in self.__inh_ct_files[:n]],      axis=0)
        self.__inh_labels_proj  = np.stack([load_numpy(f) for f in self.__inh_labels_files[:n]],  axis=0)
        self.__exh_ct_proj      = np.stack([load_numpy(f) for f in self.__exh_ct_files[:n]],      axis=0)
        self.__exh_labels_proj  = np.stack([load_numpy(f) for f in self.__exh_labels_files[:n]],  axis=0)
        self.__kv_source_angles = np.stack([load_json(f)  for f in self.__angle_files[:n]],       axis=0)
        logger.info(f'Validation data pre-loaded: inh_ct={self.__inh_ct_proj.shape} inh_labels={self.__inh_labels_proj.shape}')

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[BatchImage2D, BatchLabelImage2D, List[float]]:
        # For n_volumes=10, n_angles=10, we have 200 samples due to inhale/exhale.
        if idx < self.__n_volumes * self.__n_angles:
            volume_idx = idx // self.__n_angles
            angle_idx  = idx % self.__n_angles
            if self.__preload:
                data   = self.__inh_ct_proj[volume_idx, angle_idx]
                labels = self.__inh_labels_proj[volume_idx, angle_idx]
                angle  = self.__kv_source_angles[volume_idx, angle_idx]
            else:
                data   = load_numpy(self.__inh_ct_files[volume_idx])[angle_idx]
                labels = load_numpy(self.__inh_labels_files[volume_idx])[angle_idx]
                angle  = load_json(self.__angle_files[volume_idx])[angle_idx]
        else:
            idx        = idx - self.__n_volumes * self.__n_angles
            volume_idx = idx // self.__n_angles
            angle_idx  = idx % self.__n_angles
            if self.__preload:
                data   = self.__exh_ct_proj[volume_idx, angle_idx]
                labels = self.__exh_labels_proj[volume_idx, angle_idx]
                angle  = self.__kv_source_angles[volume_idx, angle_idx]
            else:
                data   = load_numpy(self.__exh_ct_files[volume_idx])[angle_idx]
                labels = load_numpy(self.__exh_labels_files[volume_idx])[angle_idx]
                angle  = load_json(self.__angle_files[volume_idx])[angle_idx]

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

def create_training_projections(
    dataset: TrainingDataset,
    pat: PatientID,
    projection_geometry: Dict[str, Any],
    n_angles: int = 3,
    min_angle: float = 0,
    max_angle: float = 360,
    transform: Transform | None = None,
    epoch: int | None = None,
    ) -> Tuple[BatchImage2D, BatchImage2D, BatchLabelImage2D, BatchLabelImage2D, List[float]]:
    logger.info(f"Creating projections for epoch {epoch}...")
    trainpath = os.path.join(dataset.path, 'data', 'training', pat)

    # Save projection geometry for reproducibility.
    filepath = os.path.join(trainpath, 'log', 'geometry.json')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    save_json(projection_geometry, filepath, overwrite=True)

    # Load volumes.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inh_ct, inh_affine = load_nifti(os.path.join(trainpath, 'volumes', 'inh_ct.nii.gz'))
    exh_ct, exh_affine = load_nifti(os.path.join(trainpath, 'volumes', 'exh_ct.nii.gz'))
    inh_labels = load_numpy(os.path.join(trainpath, 'volumes', 'inh_labels.npz'))
    exh_labels = load_numpy(os.path.join(trainpath, 'volumes', 'exh_labels.npz'))

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
        inh_ct, exh_ct, inh_labels, exh_labels, grid, params = transform(
            inh_ct, exh_ct, inh_labels, exh_labels,
            affine=affine, return_grid=True, return_params=True,
        )
        _, affine = grid
        if epoch is not None:
            filepath = os.path.join(trainpath, 'log', f'aug_affine_{epoch}.npz')
            np.savez(filepath, affine=affine.cpu().numpy())
            filepath = os.path.join(trainpath, 'log', f'aug_params_{epoch}.json')
            save_json(params, filepath, overwrite=True)

    # Sample kV source angles.
    kv_source_angles = list(np.random.uniform(min_angle, max_angle, n_angles))
    filepath = os.path.join(trainpath, 'log', f'kv_source_angles_{epoch}.json')
    save_json(kv_source_angles, filepath, overwrite=True)

    # Generate projections.
    inh_ct_proj, inh_labels_proj = project_ctorch(
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
    exh_ct_proj, exh_labels_proj = project_ctorch(
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
