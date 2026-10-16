from dicomset.utils import resolve_filepath, load_numpy, load_csv, create_affine
from itk import RTK as rtk
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from tqdm.auto import tqdm
from dicomset.utils import change_image_orientation, save_mha, append_row, save_csv

from mymi.utils.varian import load_xim_arc

angular_ranges = [1, 2, 5, 10, 20, 45, 90, 180, 360]

frame = 228

alpha = 0.03    # Regularisation parameter.
beta = 10       # Augmented lagrangian constraint multiplier.

# Load data.
dataset = 'VALKIM'
pat_id = 'Patient01'
fx = 'Fx2'
session = 'CBCT1'

# RTK requires (LR, SI, AP) orientation.
recon_size = (512, 87, 512)
recon_spacing = (0.908, 2.0, 0.908)
recon_origin = (-231.9992, -86.0, -231.9992)

f = r"L:\LEARN\Orange\Lung\Patient Images\PAT01\Fx2\CBCT\CBCT1\CBCT Projections\IPS\Proj 0"
data, info = load_xim_arc(f)
angles = info['kv-source-angle'].values

# Apply flat-field correction to correct for presumable bow-tie filter.
f = resolve_filepath('files:learn/lar/orange_pat01_fx02_ff_corr.npy')
ff_corr = load_numpy(f)
data = -np.log(np.clip(data, 1, None) / np.expand_dims(ff_corr, (0, 2)))

# AP frame.
c = angles[frame]

# Save all projections for reconstruction.
base_f = resolve_filepath('files:lar/fdk/projections')

# Get detector geometry for projection (MHA) files.
det_size = info.iloc[0]['det-size']
det_spacing = info.iloc[0]['det-spacing']
det_origin = (-(det_size[0] / 2) * det_spacing[0], -(det_size[1] / 2) * det_spacing[1])
det_affine = create_affine(det_spacing, det_origin)

# Save projections (MHA) files.
for i, p in tqdm(enumerate(data), total=len(data)):
    f = os.path.join(base_f, f'{i:03d}.mha')
    p, a = change_image_orientation(p, 'LS', 'RI', affine=det_affine)
    save_mha(p, a, f)

# Create geometry.xml files for reconstruction.
base_f = resolve_filepath(f'files:lar/fdk/geometries')
os.makedirs(base_f, exist_ok=True)

# Get data for each reconstruction.
def circ_diff(a, c):
    return abs((a - c + 180) % 360 - 180)

def get_close_angles(
    angles: List[float],
    central_angle: float,
    angular_range: float,
    ) -> Tuple[List[int], List[float]]:
    # return [a for a in angles if circ_diff(a)
    diffs = circ_diff(angles, central_angle)
    angle_diffs = zip(angles, diffs)
    angle_diffs = [(i, a, d) for i, (a, d) in enumerate(angle_diffs) if d <= (angular_range / 2)]
    indices = [int(ad[0]) for ad in angle_diffs]  # Required to map back to the files in the projections folder.
    angles = [float(ad[1]) for ad in angle_diffs]
    # angles = list(sorted(angles))
    return indices, angles

# Build reconstruction lists.
cols = {
    'centre-angle': float,
    'angular-range': float,
    'n-proj': int,
    'proj-indices': str,
    'proj-angles': str,
    'proj-angles-rtk': str,
}
df = pd.DataFrame(columns=cols.keys())

for r in angular_ranges:
    proj_indices, proj_angles = get_close_angles(angles, c, r)
    proj_angles_rtk = np.array(proj_angles) - c
    proj_angles_rtk = list(360 - proj_angles_rtk)
    proj_angles_rtk = [float(a) for a in proj_angles_rtk]
    df_data = {
        'centre-angle': c,
        'angular-range': r,
        'n-proj': len(proj_indices),
        'proj-indices': proj_indices,
        'proj-angles': proj_angles,
        'proj-angles-rtk': proj_angles_rtk,
    }
    df = append_row(df, df_data)

f = f'files:lar/fdk/angles.csv'
save_csv(df, f)

SID = info.iloc[0]['sid']
SDD = info.iloc[0]['sdd']
det_offset = info.iloc[0]['det-offset']

base_geo_f = resolve_filepath(f'files:lar/fdk/geometries')

for r in tqdm(angular_ranges, leave=False):
    # Save geometry.xml for reconstruction.
    row = df[(df['centre-angle'] == c) & (df['angular-range'] == r)].iloc[0]
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in tqdm(row['proj-angles-rtk'], leave=False):
        geometry.AddProjection(SID, SDD, float(a), det_offset[0], det_offset[1], 0, 0, 0, 0)
    geom_writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    f = os.path.join(base_geo_f, f"{frame}_{str(c).replace('.', '_')}_{r}.xml")
    geom_writer.SetFilename(f)
    geom_writer.SetObject(geometry)
    geom_writer.WriteFile()

# Load template.
base_log_f = resolve_filepath(f'files:lar/fdk/logs')
os.makedirs(base_log_f, exist_ok=True)
proj_f = resolve_filepath(f'files:lar/fdk/projections')
base_recon_f = resolve_filepath(f'files:lar/fdk/recons')
os.makedirs(base_recon_f, exist_ok=True)
base_script_f = resolve_filepath(f'files:lar/fdk/scripts')
os.makedirs(base_script_f, exist_ok=True)
base_geo_f = resolve_filepath(f'files:lar/fdk/geometries')
f = r"D:\Brett\code\medical-imaging\scripts\process\learn\valkim\lar\fdk_script.py"
with open(f) as f:
    template = f.read()

for r in tqdm(angular_ranges, leave=False):
    # Replace template variables
    row = df[(df['centre-angle'] == c) & (df['angular-range'] == r)].iloc[0]
    log_f = os.path.join(base_log_f, f"{frame}_{str(c).replace('.', '_')}_{r}.log")
    geo_f = os.path.join(base_geo_f, f"{frame}_{str(c).replace('.', '_')}_{r}.xml")
    recon_f = os.path.join(base_recon_f, f"{frame}_{str(c).replace('.', '_')}_{r}.mha")
    proj_regexp = f"({'|'.join([f'{i:03d}' for i in row['proj-indices']])})\.mha"
    script = template.format(
        geo_file=geo_f,
        log_file=log_f,
        out_file=recon_f,
        proj_dir=proj_f,
        proj_regexp=proj_regexp,
        recon_size=",".join([str(s) for s in recon_size]),
        recon_spacing=",".join([str(s) for s in recon_spacing]),
        recon_origin=",".join([str(s) for s in recon_origin]),
    )

    # Write script.
    f = os.path.join(base_script_f, f"script_{frame}_{str(c).replace('.', '_')}_{r}.py")
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, 'w') as ff:
        ff.write(script)

