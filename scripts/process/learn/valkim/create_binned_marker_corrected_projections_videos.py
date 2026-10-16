import dicomset as ds
from dicomset.utils import arg_to_list
import os
from tqdm import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions
from mymi.utils.io import load_numpy
from mymi.utils.plotting import play_video
from mymi.utils.projections import plot_projection

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
frames = None
filename_angles = {1: 'kv-source', 2: 'kv-source', 3: '?', 4: 'mv-source'}

for p in tqdm(pat_ids):
    pat = set.patient(f'PAT{p}')
    filename_angle = filename_angles[p]
    print(pat)

    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")
        print(fracpath)

        pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load arc data (treatment images and metadata).
            arc_data, arc_info = load_tiff_arc(fracpath, arc=a, filename_angle=filename_angle)

            # Load already-generated projection data.
            projpath = os.path.join(set.path, 'data', 'projections', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
            ct_proj = load_numpy(os.path.join(projpath, 'ct_proj_binned_corrected.npz'))
            labels_proj = load_numpy(os.path.join(projpath, 'labels_proj_binned_corrected.npz'))

            n_frames = len(arc_info)
            plot_args = [(arc_data[i], arc_info.iloc[i]) for i in range(n_frames)]
            plot_kwargs = [{'all_info': arc_info, 'drr': ct_proj[i], 'labels': labels_proj[i]} for i in range(n_frames)]
            filepath = os.path.join(projpath, f'projections_binned_marker_corrected.mp4')
            play_video(plot_projection, plot_args, plot_kwargs, n_frames=frames, savepath=filepath, show=False, overwrite=True)
