from augmed import RandomCrop, Transform
import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from dicomset.training import TrainingDataset
from dicomset.utils import hist_eq as hist_eq_fn, load_json, load_numpy
from dicomset.typing import *

def _list_proj_files(dirpath: str, prefix: str, ext: str) -> List[Tuple[int, int, str]]:
    """Return sorted (vol_idx, proj_idx, filepath) for files matching prefix/ext."""
    entries = []
    for f in os.listdir(dirpath):
        if not f.startswith(prefix) or not f.endswith(ext):
            continue
        stem = f[len(prefix):-len(ext)]
        parts = stem.split('_')
        if len(parts) == 2:
            try:
                entries.append((int(parts[0]), int(parts[1]), os.path.join(dirpath, f)))
            except ValueError:
                pass
    return sorted(entries)


def _filter_files(
    entries: List[Tuple[int, int, str]],
    n_volumes: Optional[int],
    n_angles: Optional[int],
) -> List[str]:
    if n_volumes is not None:
        entries = [(v, p, f) for v, p, f in entries if v < n_volumes]
    if n_angles is not None:
        entries = [(v, p, f) for v, p, f in entries if p < n_angles]
    return [f for _, _, f in entries]


class StaticLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        pat: PatientID,
        batch_size: int = 32,
        n_train_volumes: int | None = None,
        n_train_angles: int | None = None,
        n_val_volumes: int | None = None,
        n_val_angles: int | None = None,
        num_workers: int = 0,
        preload_train_data: bool = False,
        preload_val_data: bool = False,
        shuffle_train: bool = True,
        threshold_labels: bool = False,
        transform_train: Transform | None = None,
        ) -> Tuple[DataLoader, DataLoader]:
        set = TrainingDataset(dataset)

        # Create train loader.
        train_set = TrainingSet(
            set,
            pat,
            n_volumes=n_train_volumes,
            n_angles=n_train_angles,
            preload=preload_train_data,
            threshold_labels=threshold_labels,
            transform=transform_train,
        )
        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=num_workers, shuffle=shuffle_train)

        # Create val loader.
        val_set = ValidationDataset(
            set,
            pat,
            n_volumes=n_val_volumes,
            n_angles=n_val_angles,
            preload=preload_val_data,
            threshold_labels=threshold_labels,
        )
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=num_workers, shuffle=False)

        return train_loader, val_loader

class TrainingSet(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        pat: PatientID,
        hist_eq: bool = True,
        n_volumes: int | None = None,
        n_angles: int | None = None,
        normalise: bool = False,
        preload: bool = False,
        threshold_labels: bool = False,
        transform: Transform | None = None,
        ) -> None:
        self.__normalise = normalise
        self.__hist_eq = hist_eq
        self.__threshold_labels = threshold_labels
        self.__transform = transform
        self.__preload = preload
        self.__debug_n = 0
        self.__debug_every = 25
        input_size = 600
        output_size = 512
        self.__crop = RandomCrop(size=output_size, centre_offset=(input_size - output_size) / 2)

        dirpath = os.path.join(dataset.path, 'data', 'training', pat, 'projections')
        self.__inh_ct_files = _filter_files(_list_proj_files(dirpath, 'inh_ct_', '.npy'), n_volumes, n_angles)
        self.__inh_labels_files = _filter_files(_list_proj_files(dirpath, 'inh_labels_', '.npy'), n_volumes, n_angles)
        self.__exh_ct_files = _filter_files(_list_proj_files(dirpath, 'exh_ct_', '.npy'), n_volumes, n_angles)
        self.__exh_labels_files = _filter_files(_list_proj_files(dirpath, 'exh_labels_', '.npy'), n_volumes, n_angles)
        self.__angle_files = _filter_files(_list_proj_files(dirpath, 'angle_', '.json'), n_volumes, n_angles)

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
        print('pre-loading training data')
        self.__inh_ct_proj = np.stack([self._load_ct_file(f) for f in self.__inh_ct_files], axis=0)
        self.__inh_labels_proj = np.stack([load_numpy(f) for f in self.__inh_labels_files], axis=0)
        self.__exh_ct_proj = np.stack([self._load_ct_file(f) for f in self.__exh_ct_files], axis=0)
        self.__exh_labels_proj = np.stack([load_numpy(f) for f in self.__exh_labels_files], axis=0)
        self.__kv_source_angles = np.array([load_json(f) for f in self.__angle_files])
        print('training data loaded')
        print(self.__inh_ct_proj.shape, self.__inh_labels_proj.shape, self.__exh_ct_proj.shape, self.__exh_labels_proj.shape, self.__kv_source_angles.shape)

    def _load_ct_file(self, filepath: str) -> np.ndarray:
        load_start = time.perf_counter()
        data = load_numpy(filepath)
        load_time = time.perf_counter() - load_start

        hist_time = 0.0
        if self.__hist_eq:
            hist_start = time.perf_counter()
            data = hist_eq_fn(data)
            hist_time = time.perf_counter() - hist_start

        # if self.__debug_n < 3:
        #     print(f'[TrainingSet] load_ct_file path={os.path.basename(filepath)} load={load_time:.3f}s hist_eq={hist_time:.3f}s shape={tuple(data.shape)} dtype={data.dtype}')
        return data

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[BatchImage2D, BatchLabelImage2D, List[float]]:
        total_start = time.perf_counter()
        n = len(self.__inh_ct_files)
        if idx < n:
            if self.__preload:
                load_start = time.perf_counter()
                data = self.__inh_ct_proj[idx]
                labels = self.__inh_labels_proj[idx]
                angle = self.__kv_source_angles[idx]
                load_time = time.perf_counter() - load_start
                load_kind = 'preloaded'
            else:
                load_start = time.perf_counter()
                data = self._load_ct_file(self.__inh_ct_files[idx])
                labels = load_numpy(self.__inh_labels_files[idx])
                angle = load_json(self.__angle_files[idx])
                load_time = time.perf_counter() - load_start
                load_kind = 'on_demand'
        else:
            idx = idx - n
            if self.__preload:
                load_start = time.perf_counter()
                data = self.__exh_ct_proj[idx]
                labels = self.__exh_labels_proj[idx]
                angle = self.__kv_source_angles[idx]
                load_time = time.perf_counter() - load_start
                load_kind = 'preloaded'
            else:
                load_start = time.perf_counter()
                data = self._load_ct_file(self.__exh_ct_files[idx])
                labels = load_numpy(self.__exh_labels_files[idx])
                angle = load_json(self.__angle_files[idx])
                load_time = time.perf_counter() - load_start
                load_kind = 'on_demand'

        # Crop from 600 - 512 with jitter.
        data, labels = self.__crop(data, labels)

        torch_start = time.perf_counter()
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

        if self.__normalise:
            data = (data - torch.mean(data)) / torch.std(data)

        if self.__threshold_labels:
            labels = (labels > 0).float()
        else:
            for c in range(labels.shape[0]):
                l_min, l_max = labels[c].min(), labels[c].max()
                if l_max > l_min:
                    labels[c] = (labels[c] - l_min) / (l_max - l_min)

        data = data[None, ...]
        total_time = time.perf_counter() - total_start
        torch_time = time.perf_counter() - torch_start

        # if self.__debug_n < 3 or self.__debug_n % self.__debug_every == 0:
        #     print(f'[TrainingSet] sample idx={idx} kind={load_kind} load={load_time:.3f}s torch={torch_time:.3f}s total={total_time:.3f}s data={tuple(data.shape)} labels={tuple(labels.shape)} angle_type={type(angle).__name__}')
        self.__debug_n += 1
        return data, labels, angle

    def __len__(self):
        return len(self.__inh_ct_files) * 2  # Inhale/exhale.

class ValidationDataset(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        pat: PatientID,
        hist_eq: bool = True,
        n_volumes: int | None = None,
        n_angles: int | None = None,
        normalise: bool = False,
        preload: bool = False,
        threshold_labels: bool = False,
        ) -> None:
        self.__normalise = normalise
        self.__preload = preload
        self.__hist_eq = hist_eq
        self.__threshold_labels = threshold_labels
        self.__debug_n = 0
        self.__debug_every = 25

        dirpath = os.path.join(dataset.path, 'data', 'validation', pat, 'projections')
        self.__inh_ct_files = _filter_files(_list_proj_files(dirpath, 'inh_ct_', '.npy'), n_volumes, n_angles)
        self.__inh_labels_files = _filter_files(_list_proj_files(dirpath, 'inh_labels_', '.npy'), n_volumes, n_angles)
        self.__exh_ct_files = _filter_files(_list_proj_files(dirpath, 'exh_ct_', '.npy'), n_volumes, n_angles)
        self.__exh_labels_files = _filter_files(_list_proj_files(dirpath, 'exh_labels_', '.npy'), n_volumes, n_angles)
        self.__angle_files = _filter_files(_list_proj_files(dirpath, 'angle_', '.json'), n_volumes, n_angles)

        self.__inh_ct_proj = None
        self.__inh_labels_proj = None
        self.__exh_ct_proj = None
        self.__exh_labels_proj = None
        self.__kv_source_angles = None

        if self.__preload:
            self._preload_data()

    def _preload_data(self) -> None:
        print('pre-loading validation data')
        self.__inh_ct_proj = np.stack([self._load_ct_file(f) for f in self.__inh_ct_files], axis=0)
        self.__inh_labels_proj = np.stack([load_numpy(f) for f in self.__inh_labels_files], axis=0)
        self.__exh_ct_proj = np.stack([self._load_ct_file(f) for f in self.__exh_ct_files], axis=0)
        self.__exh_labels_proj = np.stack([load_numpy(f) for f in self.__exh_labels_files], axis=0)
        self.__kv_source_angles = np.array([load_json(f) for f in self.__angle_files])
        print('validation data loaded')
        print(self.__inh_ct_proj.shape, self.__inh_labels_proj.shape, self.__exh_ct_proj.shape, self.__exh_labels_proj.shape, self.__kv_source_angles.shape)

    def _load_ct_file(self, filepath: str) -> np.ndarray:
        load_start = time.perf_counter()
        data = load_numpy(filepath)
        load_time = time.perf_counter() - load_start

        hist_time = 0.0
        if self.__hist_eq and hist_eq_fn is not None:
            hist_start = time.perf_counter()
            data = hist_eq_fn(data)
            hist_time = time.perf_counter() - hist_start

        # if self.__debug_n < 3:
        #     print(f'[ValidationDataset] load_ct_file path={os.path.basename(filepath)} load={load_time:.3f}s hist_eq={hist_time:.3f}s shape={tuple(data.shape)} dtype={data.dtype}')
        return data

    def __getitem__(
        self,
        idx: int,
        ) -> Tuple[BatchImage2D, BatchLabelImage2D, List[float]]:
        total_start = time.perf_counter()
        n = len(self.__inh_ct_files)
        if idx < n:
            if self.__preload:
                load_start = time.perf_counter()
                data = self.__inh_ct_proj[idx]
                labels = self.__inh_labels_proj[idx]
                angle = self.__kv_source_angles[idx]
                load_time = time.perf_counter() - load_start
                load_kind = 'preloaded'
            else:
                load_start = time.perf_counter()
                data = self._load_ct_file(self.__inh_ct_files[idx])
                labels = load_numpy(self.__inh_labels_files[idx])
                angle = load_json(self.__angle_files[idx])
                load_time = time.perf_counter() - load_start
                load_kind = 'on_demand'
        else:
            idx = idx - n
            if self.__preload:
                load_start = time.perf_counter()
                data = self.__exh_ct_proj[idx]
                labels = self.__exh_labels_proj[idx]
                angle = self.__kv_source_angles[idx]
                load_time = time.perf_counter() - load_start
                load_kind = 'preloaded'
            else:
                load_start = time.perf_counter()
                data = self._load_ct_file(self.__exh_ct_files[idx])
                labels = load_numpy(self.__exh_labels_files[idx])
                angle = load_json(self.__angle_files[idx])
                load_time = time.perf_counter() - load_start
                load_kind = 'on_demand'

        if self.__normalise:
            data = (data - np.mean(data)) / np.std(data)

        if self.__threshold_labels:
            labels = (labels > 0).astype(np.float32)
        else:
            # Per-slice min-max normalise labels to [0, 1] (matching Lung codebase).
            for c in range(labels.shape[0]):
                l_min, l_max = labels[c].min(), labels[c].max()
                if l_max > l_min:
                    labels[c] = (labels[c] - l_min) / (l_max - l_min)

        # Add singleton channel dimension.
        data = data[None, ...]
        total_time = time.perf_counter() - total_start

        # if self.__debug_n < 3 or self.__debug_n % self.__debug_every == 0:
        #     print(f'[ValidationDataset] sample idx={idx} kind={load_kind} load={load_time:.3f}s total={total_time:.3f}s data={tuple(data.shape)} labels={tuple(labels.shape)}')
        self.__debug_n += 1

        return data, labels, angle

    def __len__(self):
        return len(self.__inh_ct_files) * 2  # Inhale/exhale.
