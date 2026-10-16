from augmed import Pipeline, RandomAffine
from augmed.utils import save_json
from dicomset.nifti.utils import create_dataset as create_nifti_dataset, load_dataset as load_nifti_dataset
from dicomset.typing import *
from dicomset.utils.args import arg_to_list
from dicomset.utils.conversion import to_numpy, to_tensor
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils.io import load_json, load_nifti, save_nifti, save_numpy
from dicomset.utils.logging import logger
from dicomset.nifti.utils import create_ct, create_index, create_region
from dicomset.training.utils import create_dataset as create_ds_training_dataset
import numpy as np
import os
import shutil
from skimage.restoration import inpaint_biharmonic
import torch
from tqdm.auto import tqdm
from typing import List

from mymi.transforms import pad
from mymi.utils.cdog import load_tiff
from mymi.utils.projections import blur_image, project_ctorch

def create_valkim_preprocessed_dataset(
    patient_id: PatientID | List[PatientID] = ['PAT1', 'PAT2', 'PAT3', 'PAT4'],
    recreate: bool = False,
    ) -> None:
    dataset = 'VALKIM'
    dataset_pp = 'VALKIM-PP'
    patient_ids = arg_to_list(patient_id, str)
    # Only patients 1/2 have fiducials mask.
    marker_regions = ['bc_Marker_1', 'bc_Marker_2', 'bc_Marker_3']
    phase_regions = marker_regions
    avg_regions = ['GTV_Inh', 'GTV_Exh']
    inh_series = 'series_0'
    exh_series = 'series_5'
    avg_reg_series = 'series_10'     # Give it a new number as it doesn't match any of the CT series.
    old_set = load_nifti_dataset(dataset)
    new_set = create_nifti_dataset(dataset_pp, recreate=recreate)

    # Copy index.
    index = old_set.index()
    index['dataset'] = dataset_pp
    create_index(dataset_pp, index)

    # Copy regions map.
    srcpath = os.path.join(old_set.path, 'struct_map.yaml')
    destpath = os.path.join(new_set.path, 'struct_map.yaml')
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

            # Blur inhale/exhale using marker masks from average CT.
            inhale_series = study.regions_series(inh_series) 
            _, markers_data = inhale_series.data(r=marker_regions)
            markers_data = markers_data.sum(axis=0) > 0
            inhale_ct_data = blur_image(inhale_ct_data, mask=markers_data, dilate=5)

            exhale_series = study.regions_series(exh_series) 
            _, markers_data = exhale_series.data(r=marker_regions)
            markers_data = markers_data.sum(axis=0) > 0
            exhale_ct_data = blur_image(exhale_ct_data, mask=markers_data, dilate=5)

            # Save images.
            create_ct(dataset_pp, p, s, inh_series, inhale_ct_data, inhale_ct_affine)
            create_ct(dataset_pp, p, s, exh_series, exhale_ct_data, exhale_ct_affine)

            # Load avg regions data.
            avg_regions_series = study.regions_series(avg_reg_series)
            f = avg_regions_series.dicom.filepath
            assert '/RS.000000.dcm' in f, f
            avg_regions, avg_regions_data = avg_regions_series.data(r=avg_regions, return_regions=True)

            # Copy markers and regions.
            for r, d in zip(avg_regions, avg_regions_data):
                if r == 'GTV_Inh':
                    create_region(dataset_pp, p, s, inh_series, 'GTV', d, inhale_ct_affine)
                elif r == 'GTV_Exh':
                    create_region(dataset_pp, p, s, exh_series, 'GTV', d, exhale_ct_affine)

            # Load phase-specific regions data.
            for phase in range(10):
                phase_series = study.ct_series(f'series_{phase}')
                phase_ct_affine = phase_series.affine
                phase_regions_series = study.regions_series(f'series_{phase}')
                phase_regions, phase_regions_data = phase_regions_series.data(r=phase_regions, return_regions=True)
                for r, d in zip(phase_regions, phase_regions_data):
                    create_region(dataset_pp, p, s, f'series_{phase}', r, d, phase_ct_affine)

            # Copy other images.
            # We don't need the average.
            other_series = [1, 2, 3, 4, 6, 7, 8, 9]
            for sr in other_series:
                ct_data = study.ct_series(f'series_{sr}').data
                ct_affine = study.ct_series(f'series_{sr}').affine
                assert np.all(ct_affine == inhale_ct_affine), f"Expected affine of series_{sr} to match inhale/exhale affines, got {ct_affine} and {inhale_ct_affine} for patient {p} study {s}."
                if ct_data.shape != exhale_ct_data.shape:
                    ct_data = pad(ct_data, ((0, 0, 0), exhale_ct_data.shape))
                # Load markers and blur.
                phase_regions_series = study.regions_series(f'series_{sr}')
                _, markers_data = phase_regions_series.data(r=marker_regions)
                markers_data = markers_data.sum(axis=0) > 0
                ct_data = blur_image(ct_data, mask=markers_data)
                create_ct(dataset_pp, p, s, f'series_{sr}', ct_data, ct_affine)
