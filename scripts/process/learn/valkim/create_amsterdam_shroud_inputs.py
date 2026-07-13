# The tiff files can't be read by SimpleITK, so we need to convert to an
# appropriate format.
import os
import SimpleITK as sitk
from tqdm.auto import tqdm

from mymi.utils.cdog import list_tiff_fractions, list_tiff_arcs, get_kv_path, load_tiff

pat_ids = [1]
pathpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]

for pat_id in tqdm(pat_ids, desc='Patients'):
    pat_path = os.path.join(pathpath, f'Patient{pat_id:02d}')
    frac_ids = list_tiff_fractions(pat_path) if fractions == 'all' else fractions
    for frac_id in tqdm(frac_ids, desc='Fractions', leave=False):
        frac_path = os.path.join(pat_path, f'Fx{frac_id:02d}')
        arc_ids = list_tiff_arcs(frac_path) if arcs == 'all' else arcs
        kv_path = get_kv_path(frac_path)
        dest_path = kv_path + '_AS'
        os.makedirs(dest_path, exist_ok=True)
        for arc_id in tqdm(arc_ids, desc='Arcs', leave=False):
            tiff_files = sorted(
                [f for f in os.listdir(kv_path) if f.endswith('.tiff') and f.split('_')[1] == str(arc_id)],
                key=lambda f: int(f.split('_')[2]),
            )
            arc_dir = os.path.join(dest_path, f'arc_{arc_id}')
            os.makedirs(arc_dir, exist_ok=True)
            for i, fname in enumerate(tqdm(tiff_files, desc='Frames', leave=False)):
                data, info = load_tiff(os.path.join(kv_path, fname))  # data: (LR, SI)
                img = sitk.GetImageFromArray(data)
                img.SetSpacing((info['det-spacing'][1], info['det-spacing'][0]))  # (SI_sp, LR_sp)
                sitk.WriteImage(img, os.path.join(arc_dir, f'frame_{i:05d}.mha'))

print('Done.')
