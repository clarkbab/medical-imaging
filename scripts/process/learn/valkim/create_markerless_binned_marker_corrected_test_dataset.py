import dicomset as ds
from dicomset.utils import arg_to_list, load_numpy, minmax, centre_crop, centre_crop_points
import numpy as np
import os
import pandas as pd
from PIL import Image as PILImage
from skimage.restoration import inpaint_biharmonic
from tqdm import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions
from mymi.utils.projections import blur_image

OUTPUT_SIZE = (512, 512)
PNG_MAX = 65535
PNG_DTYPE = np.uint16

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1, 2, 3]
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
            labels = load_numpy(os.path.join(projpath, 'labels_proj_binned_corrected.npz'))

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

                # Process intensity.
                d = np.log(d + 1)
                # If we don't exclude outliers, the resulting histogram can be very skewed, and
                # the image can appear very dark.
                min = np.percentile(d, 0.1)
                max = np.percentile(d, 99.9)
                d = minmax(d, data_min=min, data_max=max)
                d = (d * PNG_MAX)
                l = (l * PNG_MAX).astype(PNG_DTYPE)
                d = PNG_MAX - d  # Invert.

                # Remove markers via inpainting and add Poisson noise.
                d = blur_image(d, points=m, radius=10)
                d = minmax(d)
                d = (d * PNG_MAX).astype(PNG_DTYPE)

                total = np.concatenate([d, l], axis=0)

                # Save image.
                # Filename convention matches MATLAB generateTestDataset.m:
                # {trial[0]}{centre[0]}{patient:02d}_{fraction:02d}_{frame:04d}_{angle:.2f}.png
                # i.e. V(ALKIM) + R(NSH) + patient + fraction (not arc).
                savepath = rf"D:\Brett\data\lungtrack\patients\VALKIM_RNSH_PAT{p:02d}\mymi-test\Fx{f:02d}\Arc{a:02d}\VR{p:02d}_{f:02d}_{frame + 1:04d}_{kv_source_angle:.2f}.png"
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                total = np.flip(total, axis=1)
                total = total.T
                print(savepath)
                PILImage.fromarray(total, mode="I;16").save(savepath)
