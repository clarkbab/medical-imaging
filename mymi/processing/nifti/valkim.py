from augmed import Pipeline, RandomAffine
from augmed.utils import save_json, to_numpy, to_tensor
import numpy as np
import os
import shutil
from skimage.restoration import inpaint_biharmonic
import torch
from tqdm import tqdm

from mymi.datasets.nifti import load as load_nifti_dataset
from mymi.datasets.nifti.utils import create_ct, create_index, create_region
from mymi.datasets.training import create as create_training, exists as exists_training, load as load_training
from mymi import logging
from mymi.processing import create_ctorch_projections
from mymi.utils import load_json, load_nifti, load_tiff, from_rtplan_dicom, save_nifti, save_numpy, with_makeitso

def create_valkim_preprocessed_dataset(
    makeitso: bool = False,
    ) -> None:
    dataset = 'VALKIM'
    dataset_pp = 'VALKIM-PP'
    # Only patients 1/2 have fiducials mask.
    pat_ids = ['PAT1', 'PAT2']
    marker_regions = ['Fiducial_1', 'Fiducial_2', 'Fiducial_3']
    structure_regions = ['Carina', 'Chestwall', 'Chestwall_LT', 'Chestwall_RT', 'GreatVes', 'GTV_Inh', 'GTV_Exh', 'Heart', 'Liver', 'Lung_L', 'Lung_R', 'Nerve_Root', 'Oesophagus', 'Spinal_Cord', 'Spleen', 'Stomach']
    regions = marker_regions + structure_regions
    inh_series = 'series_0'
    exh_series = 'series_5'
    input_reg_series = 'series_0'
    avg_reg_series = 'series_10'     # Give it a new number as it doesn't match any of the CT series.
    old_set = load_nifti_dataset(dataset)

    # Copy index.
    index = old_set.index()
    index = index[index['patient-id'].isin(pat_ids)]
    create_index(dataset_pp, index, makeitso=makeitso)

    for p in tqdm(pat_ids):
        pat = old_set.patient(p)
        study_ids = pat.list_studies()
        for s in study_ids:
            study = pat.study(s)

            # Load inhale CT data.
            inhale_series = study.ct_series(inh_series)
            f = inhale_series.dicom.filepath
            assert ' 0% ' in f or ' 00%' in f, f
            inhale_ct_data = inhale_series.data
            inhale_ct_affine = inhale_series.affine

            # Load exhale CT data.
            exhale_series = study.ct_series(exh_series)
            f = exhale_series.dicom.filepath
            assert ' 50%' in f, f
            exhale_ct_data = exhale_series.data
            exhale_ct_affine = exhale_series.affine

            # Load regions data.
            regions_series = study.regions_series(input_reg_series)
            f = regions_series.dicom.filepath
            assert '/RS.000000.dcm' in f, f
            regions_data = regions_series.data(regions=regions)

            # Blur inhale/exhale using marker masks from average CT.
            markers_data = regions_series.data(regions=marker_regions)
            markers_data = markers_data.sum(axis=0) > 0
            inhale_ct_data = inpaint_biharmonic(inhale_ct_data, markers_data, split_into_regions=False)
            exhale_ct_data = inpaint_biharmonic(exhale_ct_data, markers_data, split_into_regions=False)

            # Save images.
            create_ct(dataset_pp, p, s, inh_series, inhale_ct_data, inhale_ct_affine, makeitso=makeitso)
            create_ct(dataset_pp, p, s, exh_series, exhale_ct_data, exhale_ct_affine, makeitso=makeitso)

            # Copy regions.
            assert np.all(inhale_ct_affine == exhale_ct_affine)
            for r, d in zip(regions, regions_data):
                if r == 'GTV_Inh':
                    create_region(dataset_pp, p, s, inh_series, 'GTV', d, inhale_ct_affine, makeitso=makeitso)
                elif r == 'GTV_Exh':
                    create_region(dataset_pp, p, s, exh_series, 'GTV', d, exhale_ct_affine, makeitso=makeitso)
                else:
                    create_region(dataset_pp, p, s, avg_reg_series, r, d, inhale_ct_affine, makeitso=makeitso)

def create_valkim_training_dataset(
    create_train_volumes: bool = True,
    create_val_projections: bool = True,
    create_val_volumes: bool = True,
    makeitso: bool = False,
    n_val_angles: int = 3,
    n_val_volumes: int = 3,
    ) -> None:
    logging.log_args("Creating VALKIM training dataset")
    dataset = 'VALKIM-PP'
    inh_series = 'series_0'
    exh_series = 'series_5'
    roi_series = 'series_0'
    min_angle = 0
    max_angle = 360
    nifti_set = load_nifti_dataset(dataset)
    training_set = create_training(dataset, makeitso=makeitso) if not exists_training(dataset) else load_training(dataset)
    pat_ids = nifti_set.list_patients()
    inh_regions = ['GTV_Inh', 'Lung_L', 'Lung_R']
    exh_regions = ['GTV_Exh', 'Lung_L', 'Lung_R']

    info = {
        'PAT1': {
            'rtplan': r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\Patient01\243-RT LUNG Plan\RP.000000.dcm",
            'treatment-image': r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx01\kV\Ch1_1_7357_289.97.tiff",
        },
        'PAT2': {
            'rtplan': r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\Patient02\361-LT LUNG Plan\RP.000000.dcm",
            'treatment-image': r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx01\kV\Ch1_1_4526_249.98.tiff",
        }
    }

    # Copy non-augmented volumes as the training data.
    if create_train_volumes:
        for p in tqdm(pat_ids, desc="Creating training volumes for patients"):
            # Load volumes.
            pat = nifti_set.patient(p)
            inhale_series = pat.study('study_0').ct_series(inh_series)
            exhale_series = pat.study('study_0').ct_series(exh_series)
            reg_series = pat.study('study_0').regions_series(roi_series)
            inh_ct = inhale_series.data
            inh_affine = inhale_series.affine
            exh_ct = exhale_series.data
            exh_affine = exhale_series.affine
            inh_labels, inh_label_names = reg_series.data(regions=inh_regions, return_regions=True)
            exh_labels, exh_label_names = reg_series.data(regions=exh_regions, return_regions=True)

            trainpath = os.path.join(training_set.path, 'data', 'training', p)
            volpath = os.path.join(trainpath, 'volumes')
            os.makedirs(volpath, exist_ok=True)

            # Save volumes.
            filepath = os.path.join(volpath, f"inh_ct.nii.gz")
            save_nifti(inh_ct, inh_affine, filepath)
            filepath = os.path.join(volpath, f"exh_ct.nii.gz")
            save_nifti(exh_ct, exh_affine, filepath)

            # Save labels.
            filepath = os.path.join(volpath, f"inh_labels.npy")
            save_numpy(inh_labels, filepath)
            filepath = os.path.join(volpath, f"exh_labels.npy")
            save_numpy(exh_labels, filepath)
            filepath = os.path.join(volpath, f"inh_label_names.json")
            save_json(inh_label_names, filepath)
            filepath = os.path.join(volpath, f"exh_label_names.json")
            save_json(exh_label_names, filepath)

    # Create augmentation pipeline.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = Pipeline([
        RandomAffine(r=10, s=[0.8, 1.2], t=20),
    ], device=device)
    print(pipeline)

    # === Create augmented volumes ===
    if create_val_volumes:
        for p in tqdm(pat_ids, desc="Creating augmented validation volumes for patients"):
            # Load volumes.
            pat = nifti_set.patient(p)
            inhale_series = pat.study('study_0').ct_series(inh_series)
            exhale_series = pat.study('study_0').ct_series(exh_series)
            reg_series = pat.study('study_0').regions_series(roi_series)
            inh_ct = inhale_series.data
            inh_affine = inhale_series.affine
            exh_ct = exhale_series.data
            exh_affine = exhale_series.affine
            inh_labels, inh_label_names = reg_series.data(regions=inh_regions, return_regions=True)
            exh_labels, exh_label_names = reg_series.data(regions=exh_regions, return_regions=True)

            valpath = os.path.join(training_set.path, 'data', 'validation', p)
            volpath = os.path.join(valpath, 'volumes')
            os.makedirs(volpath, exist_ok=True)

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
                # Perform transform.
                # What is the GPU behaviour?
                # Assigning each tensor to a device is tedious, a transform should accept a device (during init or call)
                # which overrides the device of the input array/tensor.
                inh_ct_t, exh_ct_t, inh_labels_t, exh_labels_t, affine_t, params = pipeline(inh_ct, exh_ct, inh_labels, exh_labels, affine=affine, return_affine=True, return_params=True)

                # Save volumes.
                inh_ct_t = to_numpy(inh_ct_t)
                exh_ct_t = to_numpy(exh_ct_t)
                affine_t = to_numpy(affine_t)
                filepath = os.path.join(volpath, f"inh_ct_{i}.nii.gz")
                save_nifti(inh_ct_t, affine_t, filepath)
                filepath = os.path.join(volpath, f"exh_ct_{i}.nii.gz")
                save_nifti(exh_ct_t, affine_t, filepath)

                # Save labels.
                inh_labels_t = to_numpy(inh_labels_t)
                exh_labels_t = to_numpy(exh_labels_t)
                filepath = os.path.join(volpath, f"inh_labels_{i}.nii.gz")
                save_nifti(inh_labels_t, affine_t, filepath)
                filepath = os.path.join(volpath, f"exh_labels_{i}.nii.gz")
                save_nifti(exh_labels_t, affine_t, filepath)
                filepath = os.path.join(volpath, f"inh_label_names_{i}.json")
                save_json(inh_label_names, filepath)
                filepath = os.path.join(volpath, f"exh_label_names_{i}.json")
                save_json(exh_label_names, filepath)

                # Save params.
                filepath = os.path.join(volpath, f"params_{i}.json")
                save_json(params, filepath)

    # === Create projections ===
    if create_val_projections:
        for p in tqdm(pat_ids, desc="Creating val projections for patients"):
            # Load treatment isocentre from planning CT.
            filepath = info[p]['rtplan']
            plan_info = from_rtplan_dicom(filepath)

            # Load other projection geometry from .tiff files.
            # Does this change between fractions??
            filepath = info[p]['treatment-image']
            _, tiff_info = load_tiff(filepath)

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
            os.makedirs(projpath, exist_ok=True)

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
                save_json(kv_source_angles, filepath)

                # Create projections.
                inh_ct_proj, inh_labels_proj = create_ctorch_projections(
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
                )

                exh_ct_proj, exh_labels_proj = create_ctorch_projections(
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
                )

                # Save the projections.
                filepath = os.path.join(projpath, f"inh_ct_{i}.npy")
                save_numpy(inh_ct_proj, filepath)
                filepath = os.path.join(projpath, f"exh_ct_{i}.npy")
                save_numpy(exh_ct_proj, filepath)
                filepath = os.path.join(projpath, f"inh_labels_{i}.npy")
                save_numpy(inh_labels_proj, filepath)
                filepath = os.path.join(projpath, f"exh_labels_{i}.npy")
                save_numpy(exh_labels_proj, filepath)
                filepath = os.path.join(projpath, f"inh_label_names_{i}.json")
                save_json(inh_label_names, filepath)
                filepath = os.path.join(projpath, f"exh_label_names_{i}.json")
                save_json(exh_label_names, filepath)
