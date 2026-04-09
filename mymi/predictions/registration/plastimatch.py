from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import numpy as np
import os
import SimpleITK as sitk
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

from dicomset.utils.logging import logger
from dicomset.utils.io import save_nifti
from mymi.transforms import sitk_save_transform

PLASTIMATCH_TEMPLATE = """\
[GLOBAL]
fixed={fixed_path}
moving={moving_path}
xform_out={transform_path}

[STAGE]
xform=bspline
optim=lbfgsb
metric=mse
regularization=analytic
regularization-lambda=1
pgtol=0.001
grid_spac=128 128 128
res=32 32 32
max_its=1000

[STAGE]
grid_spac=64 64 64
res=16 16 16

[STAGE]
grid_spac=32 32 32
res=8 8 8

[STAGE]
grid_spac=16 16 16
res=4 4 4

[STAGE]
grid_spac=8 8 8
res=2 2 2
"""

def register_plastimatch(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    container_path: str | None = None,
    template: str | None = None,
    keep_temp: bool = False,
    ) -> sitk.Transform:
    affine = fixed_affine
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)

    temp_dir = tempfile.mkdtemp()
    if keep_temp:
        logger.info(f"Temp directory: {temp_dir}")

    # Save images to temp.
    fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
    save_nifti(fixed_ct, affine, fixed_path)
    moving_path = os.path.join(temp_dir, 'moving.nii.gz')
    save_nifti(moving_ct, moving_affine, moving_path)
    transform_path = os.path.join(temp_dir, 'bspline_coef.txt')

    # Write config.
    if template is None:
        template = PLASTIMATCH_TEMPLATE
    pm_config = template.format(
        fixed_path=fixed_path,
        moving_path=moving_path,
        transform_path=transform_path,
    )
    config_path = os.path.join(temp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write(pm_config)

    # Run plastimatch.
    if container_path is not None:
        command = [
            'singularity',
            '--verbose',
            'exec',
            '--bind', f'{temp_dir}:{temp_dir}',
            container_path,
            'plastimatch', 'register',
            config_path,
        ]
    else:
        command = ['plastimatch', 'register', config_path]

    logger.info(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Plastimatch failed with return code {result.returncode}")

    # Parse B-spline coefficient file.
    if not os.path.exists(transform_path):
        raise RuntimeError(f"Plastimatch failed — coefficient file not found at '{transform_path}'.")

    with open(transform_path, 'r') as f:
        lines = f.readlines()

    spline_order = 3
    assert lines[1].startswith('img_origin = ')
    img_origin = tuple(float(o) for o in lines[1].split(' = ')[-1].strip().split())
    assert lines[2].startswith('img_spacing = ')
    img_spacing = tuple(float(s) for s in lines[2].split(' = ')[-1].strip().split())
    assert lines[3].startswith('img_dim = ')
    img_dim = tuple(float(d) for d in lines[3].split(' = ')[-1].strip().split())
    assert lines[6].startswith('vox_per_rgn = ')
    vox_per_rgn = tuple(int(v) for v in lines[6].split(' = ')[-1].strip().split())
    assert lines[7].startswith('direction_cosines = ')
    direction_cosines = tuple(float(c) for c in lines[7].split(' = ')[-1].strip().split())

    # Image sizes that are perfectly divisible by 'vox_per_rgn' should have one fewer control point.
    hack = 1e-6
    mesh_size = tuple(int(s) for s in np.floor((np.array(img_dim) / vox_per_rgn) - hack).astype(int) + 1)
    n_coefs = 3 * np.prod(np.array(mesh_size) + spline_order)
    coefs = [float(l.strip()) for l in lines[8:]]
    if len(coefs) != n_coefs:
        raise ValueError(f"Expected {n_coefs} coefficients, but found {len(coefs)} in {transform_path}.")

    # Create B-spline transform.
    # Plastimatch loads nifti with -x/y direction cosines, so wrap with affine transforms.
    transform = sitk.CompositeTransform(3)
    affine_t = sitk.AffineTransform(3)
    affine_t.SetMatrix((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    transform.AddTransform(affine_t)
    bspline = sitk.BSplineTransform(3)
    bspline.SetTransformDomainDirection(direction_cosines)
    bspline.SetTransformDomainMeshSize(mesh_size)
    bspline.SetTransformDomainOrigin(img_origin)
    bspline.SetTransformDomainPhysicalDimensions(img_dim)
    bspline.SetParameters(coefs)    # Must be set after domain.
    transform.AddTransform(bspline)
    transform.AddTransform(affine_t)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return transform
