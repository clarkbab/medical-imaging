import dicomset as ds
from dicomset.utils import arg_to_list, load_numpy, minmax, centre_crop, centre_crop_points, hist_eq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions
from mymi.utils.projections import blur_image

OUTPUT_SIZE = (512, 512)

dataset = 'VALKIM-PP'
tdataset = 'VALKIM-PP-STATIC'
set = ds.load(dataset, 'nifti')
tset = ds.load(tdataset, 'training')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1, 2, 3]
arcs = [1]
filename_angles = {1: 'kv-source', 2: 'kv-source', 3: '?', 4: 'mv-source'}

for p in tqdm(pat_ids):
    filename_angle = filename_angles[p]

    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")

        pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load arc data.
            data, info = load_tiff_arc(fracpath, arc=a, filename_angle=filename_angle, invert_intensity=False)
            n_frames = len(info)

            # Load GT labels.
            projpath = os.path.join(set.path, 'data', 'projections', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
            labels = load_numpy(os.path.join(projpath, 'labels_proj_interp_corrected.npz'))

            # Load IF markers.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient{p:02d}\Fx{f:02d}\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
            df = pd.read_csv(filepath)
            if_markers = df[['Marker_0_X', 'Marker_0_Y', 'Marker_1_X', 'Marker_1_Y', 'Marker_2_X', 'Marker_2_Y']].to_numpy()
            if_markers = if_markers.reshape((-1, 3, 2))

            # Flip markers.
            if_markers[:, :, 1] = data.shape[2] - if_markers[:, :, 1]

            for frame in tqdm(range(n_frames), desc='Processing frames', leave=False):
                d = data[frame]
                kv_source_angle = info.iloc[frame]['kv-source-angle']

                l = labels[frame, 0]    # GTV
                m = if_markers[frame]   # Markers

                # Crop images/labels.
                orig_size = d.shape
                d = centre_crop(d, OUTPUT_SIZE, dim=2)
                l = centre_crop(l, OUTPUT_SIZE, dim=2)
                m = centre_crop_points(m, OUTPUT_SIZE, orig_size)

                # Convert image from photon counts to attenuation.
                d = np.log(d + 1)
                # If we don't exclude outliers, the resulting histogram can be very skewed, and
                # the image can appear very dark.
                min = np.percentile(d, 0.1)
                max = np.percentile(d, 99.9)
                d = minmax(d, data_min=min, data_max=max)
                d = d.max() - d

                # Remove markers via inpainting.
                d = blur_image(d, points=m, radius=10)

                d = minmax(d).astype(np.float32)
                l = l.astype(np.float32)

                # Threshold the labels.
                l = np.where(l > 0.5, 1.0, 0.0)
                l = l.astype(np.bool_)

                # Save arrays.
                basename = f"{frame + 1:04d}_{kv_source_angle:.2f}"
                savedir = os.path.join(tset.path, 'data', 'test', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
                os.makedirs(savedir, exist_ok=True)
                print(os.path.join(savedir, basename))
                np.save(os.path.join(savedir, f"{basename}_image.npy"), d)
                np.save(os.path.join(savedir, f"{basename}_label.npy"), l)

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(d.T, cmap='gray', origin='lower')
                axes[0].set_title('Image')
                axes[0].axis('off')
                axes[1].imshow(l.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                axes[1].set_title('Label (GTV)')
                axes[1].axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(savedir, f'{basename}_vis.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
