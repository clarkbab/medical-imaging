import os
import shutil
import subprocess
import tempfile
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.geometry import get_centre_of_mass
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import crop_or_pad, dvf_to_sitk_transform, resample, sitk_save_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

def create_plastimatch_predictions(
    dataset: str,
    create_coefs: bool = True,
    create_moved_ct: bool = True,
    create_moved_dose: bool = True,
    fixed_study_id: StudyID = 'study_1',
    landmark_ids: Optional[LandmarkIDs] = 'all',
    lung_region: str = 'Lungs',
    model: str = 'plastimatch',
    moving_study_id: StudyID = 'study_0',
    pat_ids: PatientIDs = 'all',
    region_ids: Optional[RegionIDs] = 'all',
    save_as_labels: bool = False,
    splits: Splits = 'all',
    use_timing: bool = True) -> None:

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        print(p)
        # Timing table data.
        data = {
            'dataset': dataset,
            'patient-id': p,
            'device': 'cuda',
        }
        with timer.record(data, enabled=use_timing):
            pat = set.patient(p)
            fixed_study = pat.study(fixed_study_id)
            moving_study = pat.study(moving_study_id)

            # Check for isotropic voxel spacing.
            fixed_spacing = fixed_study.ct_spacing
            moving_spacing = moving_study.ct_spacing
            assert fixed_spacing == moving_spacing

            # Open template.
            template_filepath = os.path.join(os.path.dirname(__file__), 'template.txt')
            with open(template_filepath, 'r') as f:
                pm_config = f.read()

            # Replace template placeholders. Paths must be relative to the container home path.
            pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study_id, p, moving_study_id)
            fixed_path = fixed_study.ct_filepath
            moving_path = moving_study.ct_filepath
            moved_path = os.path.join(pred_base, 'plastimatch', f'ct.nii.gz')
            transform_path = os.path.join(pred_base, 'plastimatch', 'bspline_coef.txt')

            # Create container paths, as the prediction folder will be mapped to /data.
            sing_base = os.path.join(set.path, 'data')
            sing_mount = os.path.join(os.sep, 'data')
            sing_fixed_path = fixed_path.replace(sing_base, sing_mount)
            sing_moving_path = moving_path.replace(sing_base, sing_mount)
            sing_moved_path = moved_path.replace(sing_base, sing_mount)
            sing_transform_path = transform_path.replace(sing_base, sing_mount)

            # Replace template placeholders.
            pm_config = pm_config.replace('{fixed_path}', sing_fixed_path)
            pm_config = pm_config.replace('{moving_path}', sing_moving_path)
            pm_config = pm_config.replace('{moved_path}', sing_moved_path)
            pm_config = pm_config.replace('{transform_path}', sing_transform_path)
            pm_config_path = os.path.join(pred_base, 'plastimatch', 'config.txt')
            save_text(pm_config, pm_config_path)
            sing_pm_config_path = pm_config_path.replace(sing_base, sing_mount)
            
            # Deformable registration.
            container_path = '/config/binaries/singularity/containers/plastimatch/1.9.3/plastimatch.sif'
            command = [
                'singularity', 
                '--verbose',
                'exec',
                '--bind', f'{sing_base}:{sing_mount}',
                container_path,
                'plastimatch', 'register',
                sing_pm_config_path,
            ]
            if create_coefs:
                logging.info(command)
                subprocess.run(command)

            if save_as_labels:
                output_path = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
            else:
                pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study_id, p, moving_study_id)

            # Convert transform to SimpleITK and save.
            filepath = os.path.join(pred_base, 'plastimatch', 'bspline_coef.txt')
            with open(filepath, 'r') as f:
                lines = f.readlines()
            spline_order = 3
            assert lines[1].startswith('img_origin = ')
            img_origin = tuple(float(o) for o in lines[1].split(' = ')[-1].strip().split())
            assert img_origin == fixed_study.ct_origin
            assert lines[2].startswith('img_spacing = ')
            img_spacing = tuple(float(s) for s in lines[2].split(' = ')[-1].strip().split())
            assert img_spacing == fixed_study.ct_spacing
            assert lines[3].startswith('img_dim = ')
            img_dim = tuple(float(d) for d in lines[3].split(' = ')[-1].strip().split())
            assert img_dim == fixed_study.ct_size
            assert lines[6].startswith('vox_per_rgn = ')
            vox_per_rgn = tuple(int(v) for v in lines[6].split(' = ')[-1].strip().split())
            assert lines[7].startswith('direction_cosines = ')
            direction_cosines = tuple(float(c) for c in lines[7].split(' = ')[-1].strip().split())
            # Image sizes (e.g. 10) that are perfectly divisble by 'vox_per_rgn' (e.g. 5), should have one fewer control point (e.g. 2, not 3).
            hack = 1e-6
            mesh_size = tuple(int(s) for s in np.floor((np.array(fixed_study.ct_size) / vox_per_rgn) - hack).astype(int) + 1)
            n_coefs = 3 * np.prod(np.array(mesh_size) + spline_order)
            coefs = [float(l.strip()) for l in lines[8:]]
            if len(coefs) != n_coefs:
                raise ValueError(f"Expected {n_coefs} coefficients, but found {len(coefs)} in {filepath}.\n{lines[:8]}")

            # Create b-spline transform.
            # Plastimatch loads nifti with -x/y direction cosines, so pad with affine transforms.
            transform = sitk.CompositeTransform(3)
            affine = sitk.AffineTransform(3)
            affine.SetMatrix((-1, 0, 0, 0, -1, 0, 0, 0, 1))
            transform.AddTransform(affine)
            bspline = sitk.BSplineTransform(3)
            bspline.SetTransformDomainDirection(direction_cosines)
            bspline.SetTransformDomainMeshSize(mesh_size)
            bspline.SetTransformDomainOrigin(fixed_study.ct_origin)
            bspline.SetTransformDomainPhysicalDimensions(fixed_study.ct_size)
            bspline.SetParameters(coefs)    # Must be set after domain.
            transform.AddTransform(bspline)
            transform.AddTransform(affine)
            filepath = os.path.join(pred_base, 'dvf', f'{model}.hdf5')
            sitk_save_transform(transform, filepath)

            # Create moved image.
            if create_moved_ct:
                moved_ct = resample(moving_study.ct_data, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(pred_base, 'ct', f'{model}.nii.gz')
                save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

            if region_ids is not None:
                pat_regions = regions_to_list(region_ids, literals={ 'all': pat.list_regions })
                for r in pat_regions:
                    # Perform transform.
                    moving_label = moving_study.region_data(region_ids=r)[r]
                    moved_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                    filepath = os.path.join(pred_base, 'regions', r, f'{model}.nii.gz')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    save_nifti(moved_label, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

            if landmark_ids is not None:
                fixed_lms_df = fixed_study.landmark_data(landmark_ids=landmark_ids)
                if fixed_lms_df is not None:
                    fixed_lms = fixed_lms_df[list(range(3))].to_numpy()
                    moved_lms = sitk_transform_points(fixed_lms, transform)
                    if np.allclose(fixed_lms, moved_lms):
                        logging.warning(f"Moved points are very similar to fixed points - identity transform?")
                    moved_lms_df = fixed_lms_df.copy()
                    moved_lms_df[list(range(3))] = moved_lms
                    filepath = os.path.join(pred_base, 'landmarks', f'{model}.csv')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    save_csv(moved_lms_df, filepath, overwrite=True)

            if create_moved_dose and moving_study.has_dose:
                moving_dose = moving_study.dose_data
                moved_dose = resample(moving_dose, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(pred_base, 'dose', f'{model}.nii.gz')
                save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        timer.save(filepath)
