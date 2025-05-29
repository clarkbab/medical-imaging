import json
import os
import shutil
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.datasets import NiftiDataset
from mymi.loaders import get_holdout_split
from mymi import logging
from mymi.transforms import resample
from mymi.typing import *
from mymi.utils import *

# This is nnU-Net's v1 data format.
def convert_to_nnunet_single_region_v1(
    # Creates a single nnU-Net 'raw' dataset per region.
    dataset: str,
    first_dataset_id: int,
    region: Region,
    normalise: bool = False,
    norm_mean: Optional[float] = None,
    norm_stdev: Optional[float] = None,
    spacing: Optional[Spacing3D] = None,
    **kwargs) -> None:
    logging.arg_log('Converting NIFTI dataset to single-region NNUNET (v1)', ('dataset', 'region'), (dataset, region))

    # Check params.
    if normalise:
        assert norm_mean is not None
        assert norm_stdev is not None

    # Get regions.
    set = NiftiDataset(dataset)
    all_regions = set.list_regions()

    # Create the datasets.
    filepath = os.path.join(config.directories.datasets, 'nnunet', 'v1', 'raw', 'nnUNet_raw_data')
    region_idx = all_regions.index(region)
    dataset_id = first_dataset_id + region_idx
    dest_dataset = f"Task{dataset_id:03}_SINGLE_REGION_{region}"
    datapath = os.path.join(filepath, dest_dataset)
    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.makedirs(datapath)
    trainpath = os.path.join(datapath, 'imagesTr')
    testpath = os.path.join(datapath, 'imagesTs')
    trainpathlabels = os.path.join(datapath, 'labelsTr')
    testpathlabels = os.path.join(datapath, 'labelsTs')

    # Get train/test split.
    t, v, tst = get_holdout_split(dataset, **kwargs)
    t += v

    # Filter patients without regions.
    t = [p for p in t if set.patient(p).has_regions(region)]
    tst = [p for p in tst if set.patient(p).has_regions(region)]

    # Write training/test patients.
    pat_ids = [t, tst]
    paths = [trainpath, testpath]
    labelpaths = [trainpathlabels, testpathlabels]
    for ps, path, labelspath in zip(pat_ids, paths, labelpaths):
        for p in tqdm(ps):
            # Load sample data.
            pat = set.patient(p)
            study = pat.default_study
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            label = study.region_data(regions=region)[region]

            # Normalise CT data.
            if normalise:
                ct_data = (ct_data - norm_mean) / norm_stdev

            # Resample input.
            if spacing is not None:
                ct_data = resample(ct_data, spacing=ct_spacing, output_spacing=spacing)

            # Save input data.
            filepath = os.path.join(path, f'{p}_0000.nii.gz')   # '0000' indicates a single channel (CT only).
            save_nifti(ct_data, filepath, spacing=spacing)

            # Resample label.
            if spacing is not None:
                label = resample(label, spacing=ct_spacing, output_spacing=spacing)

            # Save data.
            filepath = os.path.join(labelspath, f'{p}.nii.gz')   # '0000' indicates a single channel (CT only).
            save_nifti(label, filepath, spacing=spacing)

    # Create 'dataset.json'.
    train_items = [{ "image": f"./imagesTr/{p}.nii.gz", "label": f"./labelsTr/{p}.nii.gz"} for p in t]
    test_items = [f"./imagesTs/{p}.nii.gz" for p in tst]
    dataset_json = {
        "modality": {
            "0": "CT",
        },
        "labels": {
            "0": "background",
            "1": region,
        },
        "numTraining": len(t),
        "numTest": len(tst),
        "training": train_items,
        "test": test_items,
    }
    filepath = os.path.join(datapath, 'dataset.json')
    with open(filepath, 'w') as f:
        json.dump(dataset_json, f)

def convert_to_nnunet_multi_region(
    # Creates a multi-region nnU-Net 'raw' dataset.
    dataset: str,
    dataset_id: int,
    normalise: bool = False,
    norm_mean: Optional[float] = None,
    norm_stdev: Optional[float] = None,
    spacing: Optional[Spacing3D] = None,
    **kwargs) -> None:
    logging.arg_log('Converting NIFTI dataset to multi-region NNUNET', ('dataset',), (dataset,))

    # Check params.
    if normalise:
        assert norm_mean is not None
        assert norm_stdev is not None

    # Get regions.
    set = NiftiDataset(dataset)
    regions = set.list_regions()

    # Create the dataset.
    filepath = os.path.join(config.directories.datasets, 'nnunet', 'raw')
    dest_dataset = f"Dataset{dataset_id:03}_MULTI_REGION"
    datapath = os.path.join(filepath, dest_dataset)
    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.makedirs(datapath)
    trainpath = os.path.join(datapath, 'imagesTr')
    testpath = os.path.join(datapath, 'imagesTs')
    trainpathlabels = os.path.join(datapath, 'labelsTr')
    testpathlabels = os.path.join(datapath, 'labelsTs')

    # Get train/test split.
    t, v, tst = get_holdout_split(dataset, **kwargs)
    t += v

    # Write training/test patients.
    pat_ids = [t, tst]
    paths = [trainpath, testpath]
    labelpaths = [trainpathlabels, testpathlabels]
    for ps, path, labelspath in zip(pat_ids, paths, labelpaths):
        for p in tqdm(ps):
            # Load sample data.
            pat = set.patient(p)
            study = pat.default_study
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            region_data = study.region_data()

            # Normalise CT data.
            if normalise:
                ct_data = (ct_data - norm_mean) / norm_stdev

            # Resample input.
            if spacing is not None:
                ct_data = resample(ct_data, spacing=ct_spacing, output_spacing=spacing)

            # Save input data.
            filepath = os.path.join(path, f'{p}_0000.nii.gz')   # '0000' indicates a single channel (CT only).
            save_nifti(ct_data, filepath, spacing=spacing)

            # Resample label.
            if spacing is not None:
                for r, d in region_data.items():
                    region_data[r] = resample(d, spacing=ct_spacing, output_spacing=spacing)

            # Save label data.
            label = np.zeros(ct_data.shape, dtype=np.int32)
            for r, d in region_data.items():
                region_class = regions.index(r) + 1
                label[d == 1] = region_class
            filepath = os.path.join(labelspath, f'{p}.nii.gz')
            save_nifti(label, filepath, spacing=spacing)

    # Create 'dataset.json'.
    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
        },
        "numTraining": len(t),
        "file_ending": ".nii.gz"
    }
    for i, r in enumerate(regions):
        dataset_json["labels"][r] = i + 1
    filepath = os.path.join(datapath, 'dataset.json')
    with open(filepath, 'w') as f:
        json.dump(dataset_json, f)

def convert_to_nnunet_single_region(
    # Creates a single nnU-Net 'raw' dataset per region.
    dataset: str,
    first_dataset_id: int,
    region: Region,
    normalise: bool = False,
    norm_mean: Optional[float] = None,
    norm_stdev: Optional[float] = None,
    spacing: Optional[Spacing3D] = None,
    **kwargs) -> None:
    logging.arg_log('Converting NIFTI dataset to single-region NNUNET', ('dataset', 'region'), (dataset, region))

    # Check params.
    if normalise:
        assert norm_mean is not None
        assert norm_stdev is not None

    # Get regions.
    set = NiftiDataset(dataset)
    all_regions = set.list_regions()

    # Create the datasets.
    filepath = os.path.join(config.directories.datasets, 'nnunet', 'raw')
    region_idx = all_regions.index(region)
    dataset_id = first_dataset_id + region_idx
    dest_dataset = f"Dataset{dataset_id:03}_SINGLE_REGION_{region}"
    datapath = os.path.join(filepath, dest_dataset)
    if os.path.exists(datapath):
        shutil.rmtree(datapath)
    os.makedirs(datapath)
    trainpath = os.path.join(datapath, 'imagesTr')
    testpath = os.path.join(datapath, 'imagesTs')
    trainpathlabels = os.path.join(datapath, 'labelsTr')
    testpathlabels = os.path.join(datapath, 'labelsTs')

    # Get train/test split.
    t, v, tst = get_holdout_split(dataset, **kwargs)
    t += v

    # Write training/test patients.
    pat_ids = [t, tst]
    paths = [trainpath, testpath]
    labelpaths = [trainpathlabels, testpathlabels]
    n_train = 0
    for ps, path, labelspath in zip(pat_ids, paths, labelpaths):
        for p in tqdm(ps):
            # Load sample data.
            pat = set.patient(p)
            if not pat.has_regions(region):
                logging.info(f"Patient {p} missing region {region}.")
                continue
            study = pat.default_study
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            label = study.region_data(regions=region)[region]

            # Normalise CT data.
            if normalise:
                ct_data = (ct_data - norm_mean) / norm_stdev

            # Resample input.
            if spacing is not None:
                ct_data = resample(ct_data, spacing=ct_spacing, output_spacing=spacing)

            # Save input data.
            filepath = os.path.join(path, f'{p}_0000.nii.gz')   # '0000' indicates a single channel (CT only).
            save_nifti(ct_data, filepath, spacing=spacing)

            # Resample label.
            if spacing is not None:
                label = resample(label, spacing=ct_spacing, output_spacing=spacing)

            # Save label data.
            filepath = os.path.join(labelspath, f'{p}.nii.gz')   # '0000' indicates a single channel (CT only).
            save_nifti(label, filepath, spacing=spacing)

            # Increment training numbers.
            if p in t:
                n_train += 1

    # Create 'dataset.json'.
    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": {
            "background": 0,
            region: 1,
        },
        "numTraining": n_train,
        "file_ending": ".nii.gz"
    }
    filepath = os.path.join(datapath, 'dataset.json')
    with open(filepath, 'w') as f:
        json.dump(dataset_json, f)
