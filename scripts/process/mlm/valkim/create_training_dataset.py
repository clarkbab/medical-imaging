from augmed import Pipeline, RandomAffine
from augmed.utils import save_json
from dicomset.nifti.utils import load_dataset as load_nifti_dataset
from dicomset.training.utils import create_dataset as create_ds_training_dataset
from dicomset.utils.conversion import to_numpy, to_tensor
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils.io import load_json, load_nifti, save_nifti, save_numpy
from dicomset.utils.logging import logger
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

_TARGET_SIZE = (512, 512)

from mymi.utils.cdog import load_raw_frame
from mymi.utils.projections import project_ctorch


def create_training_dataset(
    create_train_volumes: bool = True,
    create_val_projections: bool = True,
    create_val_volumes: bool = True,
    # Only creates two (inh/exh) un-augmented volumes. Augmentations and 
    # projections are calculated at the start of each epoch.
    n_val_volumes: int = 3,
    n_val_angles: int = 3,
    recreate: bool = True,
    ) -> None:
    logger.log_method()
    dataset = 'VALKIM-PP'
    dest_dataset = 'VALKIM-PP'
    inh_series = 'series_0'
    exh_series = 'series_5'
    min_angle = 0
    max_angle = 360
    nifti_set = load_nifti_dataset(dataset)
    training_set = create_ds_training_dataset(dest_dataset, recreate=recreate)
    pat_ids = nifti_set.list_patients()
    pat_ids = pat_ids[:2]
    regions = ['GTV', 'ts_Lung']
    info = nifti_set.params['patient-info']

    # Copy non-augmented volumes as the training data.
    if create_train_volumes:
        for p in tqdm(pat_ids, desc="Creating training volumes for patients"):
            # Load volumes.
            pat = nifti_set.patient(p)
            inhale_series = pat.study('study_0').ct_series(inh_series)
            exhale_series = pat.study('study_0').ct_series(exh_series)
            inh_reg_series = pat.study('study_0').regions_series(inh_series)
            exh_reg_series = pat.study('study_0').regions_series(exh_series)
            inh_ct = inhale_series.data
            inh_affine = inhale_series.affine
            exh_ct = exhale_series.data
            exh_affine = exhale_series.affine
            inh_labels, inh_label_names = inh_reg_series.data(r=regions, return_regions=True)
            exh_labels, exh_label_names = exh_reg_series.data(r=regions, return_regions=True)

            trainpath = os.path.join(training_set.path, 'data', 'training', p)
            volpath = os.path.join(trainpath, 'volumes')
            if os.path.exists(volpath):
                shutil.rmtree(volpath)
            os.makedirs(volpath)

            # Save volumes.
            filepath = os.path.join(volpath, f"inh_ct.nii.gz")
            save_nifti(inh_ct, inh_affine, filepath, overwrite=True)
            filepath = os.path.join(volpath, f"exh_ct.nii.gz")
            save_nifti(exh_ct, exh_affine, filepath, overwrite=True)

            # Save labels.
            filepath = os.path.join(volpath, f"inh_labels.npy")
            save_numpy(inh_labels, filepath, overwrite=True)
            filepath = os.path.join(volpath, f"exh_labels.npy")
            save_numpy(exh_labels, filepath, overwrite=True)
            filepath = os.path.join(volpath, f"inh_label_names.json")
            save_json(inh_label_names, filepath, overwrite=True)
            filepath = os.path.join(volpath, f"exh_label_names.json")
            save_json(exh_label_names, filepath, overwrite=True)

    # Create augmentation pipeline.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = Pipeline([
        RandomAffine(r=10, s=[0.8, 1.2], t=20),
    ], device=device)
    logger.info(f"Using pipeline: {pipeline}")

    # === Create augmented volumes ===
    if create_val_volumes:
        for p in tqdm(pat_ids, desc="Creating augmented validation volumes for patients"):
            # Load volumes.
            pat = nifti_set.patient(p)
            inhale_series = pat.study('study_0').ct_series(inh_series)
            exhale_series = pat.study('study_0').ct_series(exh_series)
            inh_reg_series = pat.study('study_0').regions_series(inh_series)
            exh_reg_series = pat.study('study_0').regions_series(exh_series)
            inh_ct = inhale_series.data
            inh_affine = inhale_series.affine
            exh_ct = exhale_series.data
            exh_affine = exhale_series.affine
            inh_labels, inh_label_names = inh_reg_series.data(r=regions, return_regions=True)
            exh_labels, exh_label_names = exh_reg_series.data(r=regions, return_regions=True)

            valpath = os.path.join(training_set.path, 'data', 'validation', p)
            volpath = os.path.join(valpath, 'volumes')
            if os.path.exists(volpath):
                shutil.rmtree(volpath)
            os.makedirs(volpath)

            # Speed up the transforms.
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            inh_ct = to_tensor(inh_ct, device=device)
            exh_ct = to_tensor(exh_ct, device=device)
            inh_labels = to_tensor(inh_labels, device=device)
            exh_labels = to_tensor(exh_labels, device=device)

            assert np.all(inh_affine == exh_affine)
            affine = inh_affine

            # Transform inhale/exhale volumes.
            for i in tqdm(range(n_val_volumes), leave=False):
                inh_ct_t, exh_ct_t, inh_labels_t, exh_labels_t, grid_t, params = pipeline(inh_ct, exh_ct, inh_labels, exh_labels, affine=affine, return_grid=True, return_params=True)

                # Save volumes.
                inh_ct_t = to_numpy(inh_ct_t)
                exh_ct_t = to_numpy(exh_ct_t)
                _, affine_t = grid_t
                affine_t = to_numpy(affine_t)
                filepath = os.path.join(volpath, f"inh_ct_{i}.nii.gz")
                save_nifti(inh_ct_t, affine_t, filepath, overwrite=True)
                filepath = os.path.join(volpath, f"exh_ct_{i}.nii.gz")
                save_nifti(exh_ct_t, affine_t, filepath, overwrite=True)

                # Save labels.
                inh_labels_t = to_numpy(inh_labels_t)
                exh_labels_t = to_numpy(exh_labels_t)
                filepath = os.path.join(volpath, f"inh_labels_{i}.nii.gz")
                save_nifti(inh_labels_t, affine_t, filepath, overwrite=True)
                filepath = os.path.join(volpath, f"exh_labels_{i}.nii.gz")
                save_nifti(exh_labels_t, affine_t, filepath, overwrite=True)
                filepath = os.path.join(volpath, f"inh_label_names_{i}.json")
                save_json(inh_label_names, filepath, overwrite=True)
                filepath = os.path.join(volpath, f"exh_label_names_{i}.json")
                save_json(exh_label_names, filepath, overwrite=True)

                # Save params.
                filepath = os.path.join(volpath, f"params_{i}.json")
                save_json(params, filepath, overwrite=True)

    # === Create projections ===
    if create_val_projections:
        for p in tqdm(pat_ids, desc="Creating val projections for patients"):
            # Load treatment isocentre from planning CT.
            filepath = info[p]['rtplan']
            plan_info = from_rtplan_dicom(filepath)

            # Load other projection geometry from .tiff files.
            filepath = info[p]['treatment-image']
            _, tiff_info = load_raw_frame(filepath)

            # Set projection parameters.
            isocentre = plan_info['isocentre']
            sid = tiff_info['sid']
            sdd = tiff_info['sdd']
            det_size = tiff_info['det-size']
            det_spacing = tiff_info['det-spacing']
            det_offset = tiff_info['det-offset']
            print(isocentre, sid, sdd, det_size, det_spacing, det_offset)

            valpath = os.path.join(training_set.path, 'data', 'validation', p)
            volpath = os.path.join(valpath, 'volumes')
            projpath = os.path.join(valpath, 'projections')
            if os.path.exists(projpath):
                shutil.rmtree(projpath)
            os.makedirs(projpath)

            for i in range(n_val_volumes):
                # Load inhale/exhale volumes and labels.
                filepath = os.path.join(volpath, f"inh_ct_{i}.nii.gz")
                inh_ct, affine = load_nifti(filepath)
                filepath = os.path.join(volpath, f"exh_ct_{i}.nii.gz")
                exh_ct, _ = load_nifti(filepath)
                filepath = os.path.join(volpath, f"inh_labels_{i}.nii.gz")
                inh_labels, _ = load_nifti(filepath)
                filepath = os.path.join(volpath, f"exh_labels_{i}.nii.gz")
                exh_labels, _ = load_nifti(filepath)
                filepath = os.path.join(volpath, f"inh_label_names_{i}.json")
                inh_label_names = load_json(filepath)
                filepath = os.path.join(volpath, f"exh_label_names_{i}.json")
                exh_label_names = load_json(filepath)

                # Sample kV source angles from a uniform distribution.
                kv_source_angles = list(np.random.uniform(min_angle, max_angle, n_val_angles))
                filepath = os.path.join(projpath, f"angles_{i}.json")
                save_json(kv_source_angles, filepath, overwrite=True)

                # Create projections.
                inh_ct_proj, inh_labels_proj = project_ctorch(
                    inh_ct.astype(np.float32),
                    affine.astype(np.float32),
                    isocentre,
                    sid,
                    sdd,
                    det_size,
                    det_spacing,
                    det_offset,
                    kv_source_angles,
                    labels=inh_labels.astype(np.float32),
                    threshold_labels=False,
                )
                inh_ct_proj = F.interpolate(inh_ct_proj.unsqueeze(1).float(), size=_TARGET_SIZE, mode='bilinear', align_corners=False).squeeze(1)
                inh_labels_proj = F.interpolate(inh_labels_proj.float(), size=_TARGET_SIZE, mode='nearest')

                exh_ct_proj, exh_labels_proj = project_ctorch(
                    exh_ct.astype(np.float32),
                    affine.astype(np.float32),
                    isocentre,
                    sid,
                    sdd,
                    det_size,
                    det_spacing,
                    det_offset,
                    kv_source_angles,
                    labels=exh_labels.astype(np.float32),
                    threshold_labels=False,
                )
                exh_ct_proj = F.interpolate(exh_ct_proj.unsqueeze(1).float(), size=_TARGET_SIZE, mode='bilinear', align_corners=False).squeeze(1)
                exh_labels_proj = F.interpolate(exh_labels_proj.float(), size=_TARGET_SIZE, mode='nearest')

                # Save the projections.
                filepath = os.path.join(projpath, f"inh_ct_{i}.npy")
                save_numpy(inh_ct_proj, filepath, overwrite=True)
                filepath = os.path.join(projpath, f"exh_ct_{i}.npy")
                save_numpy(exh_ct_proj, filepath, overwrite=True)
                filepath = os.path.join(projpath, f"inh_labels_{i}.npy")
                save_numpy(inh_labels_proj, filepath, overwrite=True)
                filepath = os.path.join(projpath, f"exh_labels_{i}.npy")
                save_numpy(exh_labels_proj, filepath, overwrite=True)
                filepath = os.path.join(projpath, f"inh_label_names_{i}.json")
                save_json(inh_label_names, filepath, overwrite=True)
                filepath = os.path.join(projpath, f"exh_label_names_{i}.json")
                save_json(exh_label_names, filepath, overwrite=True)


if __name__ == '__main__':
    create_training_dataset(
        create_train_volumes=True,
        create_val_volumes=True,
        create_val_projections=True,
        n_val_angles=100,
        n_val_volumes=10,
        recreate=True,
    )
