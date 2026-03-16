from augmed import Transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *

from mymi import datasets as ds
from mymi.datasets.training import TrainingDataset
from mymi import logging
from mymi.typing import *
from mymi.utils import *
from augmed.utils import save_json, to_tensor

from .random_sampler import RandomSampler

class DRRLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        pat: PatientID,
        batch_size: int = 32,
        n_train_angles: int = 3,
        n_val_angles: int = 3,
        n_val_volumes: int = 3,
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
        # Need 'num_workers=0' as we're computing projections at the start of each epoch in the main process.
        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=0, shuffle=True)

        # Create val loader.
        val_set = ValidationDataset(
            set,
            pat,
            n_angles=n_val_angles,
            n_volumes=n_val_volumes,
        )
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=0, shuffle=False)

        return train_loader, val_loader

class TrainingSet(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        pat: PatientID,
        min_angle: float = 0,
        max_angle: float = 360,
        n_angles: int = 3,
        projection_geometry: Dict[str, Any] | None = None,
        standardise: bool = True,
        transform: Transform | None = None,
        **kwargs,
        ) -> None:
        self.__dataset = dataset
        self.__pat = pat
        self.__min_angle = min_angle
        self.__max_angle = max_angle
        self.__n_angles = n_angles
        self.__projection_geometry = projection_geometry
        self.__standardise = standardise
        self.__transform = transform
        if self.__transform is not None:
            assert affine is not None, 'Affine is required when transform applied to dataloader.'

        # Save the projection parameters.
        self.__trainpath = os.path.join(dataset.path, 'data', 'training', pat)
        filepath = os.path.join(self.__trainpath, 'log', f"geometry.json")
        save_json(projection_geometry, filepath)

        # Keep inhale/exhale volumes in memory.
        inh_ct_path = os.path.join(self.__trainpath, 'volumes', 'inh_ct.nii.gz')
        self.__inh_ct, self.__inh_affine = load_nifti(inh_ct_path)
        exh_ct_path = os.path.join(self.__trainpath, 'volumes', 'exh_ct.nii.gz')
        self.__exh_ct, self.__exh_affine = load_nifti(exh_ct_path)
        inh_labels_path = os.path.join(self.__trainpath, 'volumes', 'inh_labels.npy')
        self.__inh_labels = load_numpy(inh_labels_path)
        exh_labels_path = os.path.join(self.__trainpath, 'volumes', 'exh_labels.npy')
        self.__exh_labels = load_numpy(exh_labels_path)

    # To be called at epoch start to augment volume and generate projections.
    def create_projections(
        self,
        epoch: int | None = None,
    ) -> None:
        # Speed up the transforms.
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        inh_ct = to_tensor(self.__inh_ct, device=device)
        exh_ct = to_tensor(self.__exh_ct, device=device)
        inh_affine = to_tensor(self.__inh_affine, device=device)
        exh_affine = to_tensor(self.__exh_affine, device=device)
        inh_labels = to_tensor(self.__inh_labels, device=device)
        exh_labels = to_tensor(self.__exh_labels, device=device)

        assert torch.all(inh_affine == exh_affine), "Inhale/exhale affines do not match."
        affine = inh_affine

        # Augment the volumes.
        inh_ct_t, exh_ct_t, inh_labels_t, exh_labels_t, affine_t, params = self.__transform(inh_ct, exh_ct, inh_labels, exh_labels, affine=affine, return_affine=True, return_params=True)
        inh_ct_affine_t = affine_t
        exh_ct_affine_t = affine_t
        assert torch.all(inh_ct_affine_t == affine), "Inhale affine was modified by the transform. This should not happen as the transform should only apply spatial transforms to the data, not the affine. Check the transform implementation."
        assert torch.all(exh_ct_affine_t == affine), "Exhale affine was modified by the transform. This should not happen as the transform should only apply spatial transforms to the data, not the affine. Check the transform implementation."

        # Log the augmentation params.
        if epoch is not None:
            filepath = os.path.join(self.__trainpath, 'log', f"aug_params_{epoch}.json")
            save_json(params, filepath)

        # Sample kV source angles from a uniform distribution.
        kv_source_angles = list(np.random.uniform(self.__min_angle, self.__max_angle, self.__n_angles))
        self.__kv_source_angles = kv_source_angles
        filepath = os.path.join(self.__trainpath, 'log', f"kv_source_angles_{epoch}.json")
        save_json(kv_source_angles, filepath)

        # Create projections.
        inh_ct_proj, inh_labels_proj = create_projections(
            inh_ct_t,
            inh_ct_affine_t,
            self.__projection_geometry['isocentre'],
            self.__projection_geometry['sid'],
            self.__projection_geometry['sdd'],
            self.__projection_geometry['det_size'],
            self.__projection_geometry['det_spacing'],
            self.__projection_geometry['det_offset'],
            kv_source_angles,
            labels=inh_labels_t,
        )
        exh_ct_proj, exh_labels_proj = create_projections(
            exh_ct_t,
            exh_ct_affine_t,
            self.__projection_geometry['isocentre'],
            self.__projection_geometry['sid'],
            self.__projection_geometry['sdd'],
            self.__projection_geometry['det_size'],
            self.__projection_geometry['det_spacing'],
            self.__projection_geometry['det_offset'],
            kv_source_angles,
            labels=exh_labels_t,
        )

        self.__inh_ct_proj = inh_ct_proj
        self.__exh_ct_proj = exh_ct_proj
        self.__inh_labels_proj = inh_labels_proj
        self.__exh_labels_proj = exh_labels_proj

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[ProjImage, ProjLabelImage, Angle]:
        if not has_private_attr(self, '__inh_ct_proj') or not has_private_attr(self, '__exh_ct_proj'):
            raise ValueError("Projections have not been created. Call create_projections(epoch) at the start of the epoch.")

        # For n_angles=10, we have 20 samples due to inhale/exhale.
        if idx < self.__n_angles:
            # Get inhale image/labels.
            angle_idx = idx
            data = self.__inh_ct_proj[angle_idx]
            labels = self.__inh_labels_proj[angle_idx]
            angle = self.__kv_source_angles[angle_idx]
        else:
            # Get exhale image/labels.
            angle_idx = idx - self.__n_angles
            data = self.__exh_ct_proj[angle_idx]
            labels = self.__exh_labels_proj[angle_idx]
            angle = self.__kv_source_angles[angle_idx]

        if self.__standardise:
            data = (data - np.mean(data)) / np.std(data)

        # Add singleton channel dimension.
        data = data[None, ...]

        # Add background class.
        label_bkg = 1 - np.expand_dims(labels.sum(axis=0).clip(0, 1), axis=0)
        labels = np.concatenate([label_bkg, labels], axis=0)

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
