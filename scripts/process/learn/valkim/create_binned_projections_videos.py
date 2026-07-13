import dicomset as ds
from dicomset.utils import arg_to_list
import os
from tqdm import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions
from mymi.utils.io import load_numpy
from mymi.utils.plotting import plot_gif
from mymi.utils.projections import plot_projection

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = ['PAT1', 'PAT2', 'PAT3', 'PAT4']
pat_ids = ['PAT1']
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
frames = None
filename_angles = ['kv-source', 'kv-source', '?', 'mv-source']
pat_ids_valkim = [f'Patient{int(p.replace("PAT", "")):02d}' for p in pat_ids]

for p in tqdm(range(len(pat_ids))):
    pat_id = pat_ids[p]
    pat = set.patient(pat_id)
    filename_angle = filename_angles[p]
    print(pat)

    pat_id_valkim = pat_ids_valkim[p]
    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\{pat_id_valkim}"
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
            projpath = os.path.join(set.path, 'data', 'projections', pat_id, f'Fx{f:02d}', f'Arc{a:02d}')
            ct_proj = load_numpy(os.path.join(projpath, 'ct_proj_binned.npz'))
            labels_proj = load_numpy(os.path.join(projpath, 'labels_proj_binned.npz'))

            n_frames = len(arc_info)
            plot_args = [(arc_data[i], arc_info.iloc[i]) for i in range(n_frames)]
            plot_kwargs = [{'all_info': arc_info, 'drr': ct_proj[i], 'labels': labels_proj[i]} for i in range(n_frames)]
            filepath = os.path.join(projpath, f'projections_binned.apng')
            plot_gif(plot_projection, plot_args, plot_kwargs, n_frames=frames, savepath=filepath, show=False, overwrite=True)
