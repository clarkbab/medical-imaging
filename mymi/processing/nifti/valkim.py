from augmed import Pipeline, RandomAffine
from dicomset.nifti.utils import create_dataset as create_nifti_dataset, load_dataset as load_nifti_dataset
from dicomset.typing import *
from dicomset.utils.args import arg_to_list
from dicomset.utils.conversion import to_numpy, to_tensor
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils.io import load_nifti, save_json, save_nifti, save_numpy
from dicomset.utils.logging import logger
from dicomset.nifti.utils import create_ct, create_index, create_region
from dicomset.training.utils import create_dataset as create_training_dataset
import numpy as np
import os
import shutil
from skimage.restoration import inpaint_biharmonic
import torch
from tqdm import tqdm
from typing import List

from mymi.processing.projections import create_ctorch_projections
from mymi.transforms import pad
from mymi.utils.cdog import load_tiff

def create_valkim_preprocessed_dataset(
    blur_markers: bool = True,
    patient_id: PatientID | List[PatientID] = ['PAT1', 'PAT2', 'PAT3'],
    recreate_dataset: bool = False,
    ) -> None:
    dataset = 'VALKIM'
    dataset_pp = 'VALKIM-PP'
    patient_ids = arg_to_list(patient_id, str)
    # Only patients 1/2 have fiducials mask.
    marker_regions = ['Fiducial_1', 'Fiducial_2', 'Fiducial_3']
    structure_regions = ['Carina', 'Chestwall', 'Chestwall_LT', 'Chestwall_RT', 'GreatVes', 'GTV_Inh', 'GTV_Exh', 'Heart', 'Liver', 'Lung_L', 'Lung_R', 'Nerve_Root', 'Oesophagus', 'Spinal_Cord', 'Spleen', 'Stomach']
    regions = marker_regions + structure_regions
    inh_series = 'series_0'
    exh_series = 'series_5'
    avg_reg_series = 'series_10'     # Give it a new number as it doesn't match any of the CT series.
    old_set = load_nifti_dataset(dataset)
    new_set = create_nifti_dataset(dataset_pp, recreate=recreate_dataset)

    # Copy index.
    create_index(dataset_pp, old_set.index())

    # Copy regions map.
    srcpath = os.path.join(old_set.path, 'regions_map.yaml')
    destpath = os.path.join(new_set.path, 'regions_map.yaml')
    shutil.copy(srcpath, destpath)

    for p in tqdm(patient_ids):
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
            assert np.all(inhale_ct_affine == exhale_ct_affine), f"Expected inhale/exhale affines to match, got {inhale_ct_affine} and {exhale_ct_affine} for patient {p} study {s}."

            # Load regions data.
            regions_series = study.regions_series(avg_reg_series)
            f = regions_series.dicom.filepath
            assert '/RS.000000.dcm' in f, f
            print(regions)
            regions_data, regions_ids = regions_series.data(r=regions, return_regions=True)
            print(regions_ids)
            print(regions_data.shape)

            # Blur inhale/exhale using marker masks from average CT.
            if blur_markers:
                markers_data = regions_series.data(r=marker_regions)
                markers_data = markers_data.sum(axis=0) > 0
                inhale_ct_data = inpaint_biharmonic(inhale_ct_data, markers_data, split_into_regions=False)
                exhale_ct_data = inpaint_biharmonic(exhale_ct_data, markers_data, split_into_regions=False)

            # Save images.
            create_ct(dataset_pp, p, s, inh_series, inhale_ct_data, inhale_ct_affine)
            create_ct(dataset_pp, p, s, exh_series, exhale_ct_data, exhale_ct_affine)

            # Copy regions.
            for r, d in zip(regions_ids, regions_data):
                if r == 'GTV_Inh':
                    create_region(dataset_pp, p, s, inh_series, 'GTV', d, inhale_ct_affine)
                elif r == 'GTV_Exh':
                    create_region(dataset_pp, p, s, exh_series, 'GTV', d, exhale_ct_affine)
                else:
                    create_region(dataset_pp, p, s, avg_reg_series, r, d, inhale_ct_affine)

            # Copy other images.
            other_series = [1, 2, 3, 4, 6, 7, 8, 9, 10]
            for sr in other_series:
                ct_data = study.ct_series(f'series_{sr}').data
                ct_affine = study.ct_series(f'series_{sr}').affine
                assert np.all(ct_affine == inhale_ct_affine), f"Expected affine of series_{sr} to match inhale/exhale affines, got {ct_affine} and {inhale_ct_affine} for patient {p} study {s}."
                if ct_data.shape != exhale_ct_data.shape:
                    ct_data = pad(ct_data, ((0, 0, 0), exhale_ct_data.shape))
                if blur_markers:
                    ct_data = inpaint_biharmonic(ct_data, markers_data, split_into_regions=False)
                ct_affine = study.ct_series(f'series_{sr}').affine
                create_ct(dataset_pp, p, s, f'series_{sr}', ct_data, ct_affine)

def create_valkim_training_dataset(
    create_train_volumes: bool = True,
    create_val_projections: bool = True,
    create_val_volumes: bool = True,
    makeitso: bool = False,
    n_val_angles: int = 3,
    n_val_volumes: int = 3,
    recreate_dataset: bool = False,
    ) -> None:
    logger.log_args("Creating VALKIM training dataset")
    dataset = 'VALKIM-PP'
    inh_series = 'series_0'
    exh_series = 'series_5'
    roi_series = 'series_0'
    min_angle = 0
    max_angle = 360
    nifti_set = load_nifti_dataset(dataset)
    training_set = create_training_dataset(dataset, makeitso=makeitso, recreate=recreate_dataset)
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
            inh_labels, inh_label_names = reg_series.data(r=inh_regions, return_regions=True)
            exh_labels, exh_label_names = reg_series.data(r=exh_regions, return_regions=True)

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
            inh_labels, inh_label_names = reg_series.data(r=inh_regions, return_regions=True)
            exh_labels, exh_label_names = reg_series.data(r=exh_regions, return_regions=True)

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
