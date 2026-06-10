import dicomset as ds
from dicomset.utils import arg_to_list, plot_slice
import io
import matplotlib.pyplot as plt
import os
import PIL
import seaborn as sns
from tqdm import tqdm

from mymi.utils.cdog import load_arc, list_arcs, list_fractions, to_4dct_phase
from mymi.utils.io import load_numpy
from mymi.utils.plotting import plot_gif

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = ['PAT1', 'PAT2', 'PAT3', 'PAT4']
pat_ids = ['PAT1']
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
filename_angles = ['kv-source', 'kv-source', '?', 'mv-source']
overwrite = False
pat_ids_valkim = [f'Patient{int(p.replace("PAT", "")):02d}' for p in pat_ids]

for p in tqdm(range(len(pat_ids))):
    pat_id = pat_ids[p]
    pat = set.patient(pat_id)
    filename_angle = filename_angles[p]
    print(pat)

    pat_id_valkim = pat_ids_valkim[p]
    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\{pat_id_valkim}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': list_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")
        print(fracpath)

        pat_arcs = arg_to_list(arcs, int, literals={'all': list_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load arc data (treatment images and metadata).
            arc_data, arc_info = load_arc(fracpath, arc=a, filename_angle=filename_angle)
            kv_source_angles = arc_info['kv-source-angle'].values
            breathing_phases = arc_info['MMPhase0'].values
            phases_4dct = [sum(to_4dct_phase(p)) for p in breathing_phases]

            # Load already-generated projection data.
            projpath = os.path.join(set.path, 'data', 'projections', pat_id, f'Fx{f:02d}')
            ct_proj = load_numpy(os.path.join(projpath, 'ct_proj.npz'))
            labels_proj = load_numpy(os.path.join(projpath, 'labels_proj.npz'))

            def plot_proj(angle_idx: int) -> PIL.Image.Image:
                info = arc_info.iloc[angle_idx]
                other_info = [arc_info.iloc[i] for i in range(len(arc_info)) if i != angle_idx]

                fig = plt.figure(figsize=(16, 6))
                gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1.5])
                treatment_ax = fig.add_subplot(gs[:, 0])
                proj_ax = fig.add_subplot(gs[:, 1])
                amp_ax = fig.add_subplot(gs[0, 2])
                phase_ax = fig.add_subplot(gs[1, 2])

                labels = labels_proj[angle_idx][0]
                treatment_data = arc_data[angle_idx]
                proj_data = ct_proj[angle_idx]
                title = f"kV source: {info['kv-source-angle']:.2f}, Phase: {info['MMPhase0']:.2f}"
                plot_slice(treatment_data, ax=treatment_ax, title=title, labels=labels, window='l:0', show_labels=False, orientation='LI')
                title = f"Phase (4DCT): {phases_4dct[angle_idx]:.1f}"
                plot_slice(proj_data, ax=proj_ax, labels=labels, title=title, window='l:0', orientation='LI')

                cb_palette = sns.color_palette('colorblind')
                if other_info:
                    base_color = cb_palette[1]
                    cmap = sns.light_palette(base_color, as_cmap=True)
                    n = len(other_info)
                    for idx, i in enumerate(other_info):
                        color = cmap((idx + 1) / (n + 1))
                        amp_ax.scatter(i['kv-source-angle'], i['MMAmplitude0'], color=color, zorder=0)
                        phase_ax.scatter(i['kv-source-angle'], i['MMPhase0'], color=color, zorder=0)

                amp_ax.scatter(info['kv-source-angle'], info['MMAmplitude0'], color=cb_palette[0], zorder=1)
                phase_ax.scatter(info['kv-source-angle'], info['MMPhase0'], color=cb_palette[0], zorder=1)
                amp_ax.set_xlim(0, 360)
                amp_ax.set_ylabel('MMAmplitude0')
                amp_ax.set_title('Amplitude')
                phase_ax.set_xlim(0, 360)
                phase_ax.set_ylim(0, 360)
                phase_ax.set_xlabel('kV source angle [degrees]')
                phase_ax.set_ylabel('MMPhase0')
                phase_ax.set_title('Phase')

                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                return PIL.Image.open(buf).copy()

            angle_idxs = list(range(len(kv_source_angles)))
            plot_args = list(zip(angle_idxs))
            filepath = os.path.join(projpath, f'arc_{a}.apng')
            plot_gif(plot_proj, plot_args, savepath=filepath, overwrite=overwrite)
