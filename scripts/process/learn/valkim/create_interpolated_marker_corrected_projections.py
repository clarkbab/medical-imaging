from dicomset.utils import arg_to_list, load_numpy, save_numpy, to_numpy, centre_of_mass, resample, logger
from itertools import islice
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions

pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]

n_frames = None
n_frames = 5
labels_output = 'density'

for p in tqdm(pat_ids):
    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': lambda: list_tiff_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")

        pat_arcs = arg_to_list(arcs, int, literals={'all': lambda: list_tiff_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load IF markers.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient{p:02d}\Fx{f:02d}\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
            df = pd.read_csv(filepath)
            if_markers = df[['Marker_0_X', 'Marker_0_Y', 'Marker_1_X', 'Marker_1_Y', 'Marker_2_X', 'Marker_2_Y']].to_numpy()
            if_markers = if_markers.reshape((-1, 3, 2))

            # Load IF images.
            frames, info = load_tiff_arc(fracpath, arc=a)

            # Load projected CT.
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\ct_proj_interp.npz"
            ct = load_numpy(filepath)

            # Load projected labels.
            label_name = 'labels_proj_interp.npz' if labels_output == 'boolean' else 'labels_proj_interp_raw.npz'
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\{label_name}"
            labels = load_numpy(filepath)
            gtv = labels[:, 0]

            # Flip markers.
            if_markers[:, :, 1] = frames.shape[2] - if_markers[:, :, 1]

            # Plot.
            moved_cts = []
            moved_labels = []
            for frame in tqdm(islice(range(len(frames)), n_frames), desc='Correcting frames', leave=False):
                ifm = if_markers[frame]
                proj_markers = labels[:, 1:4]
                
                # Get source (a) and dest (b) points.
                moving_points = to_numpy(centre_of_mass(proj_markers[frame], dim=2)).astype(np.float64)
                fixed_points = if_markers[frame].astype(np.float64)
                
                # Flatten to list of lists.
                moving_points = moving_points.flatten()
                fixed_points = fixed_points.flatten()
                
                # Fit rigid transform.
                transform = sitk.LandmarkBasedTransformInitializer(
                    sitk.Euler2DTransform(),
                    fixedLandmarks=fixed_points,
                    movingLandmarks=moving_points
                )
                transform = sitk.Euler2DTransform(transform)
                
                # Resample the labels.
                moved_ct = resample(ct[frame], dim=2, transform=transform)
                moved_label = resample(labels[frame], dim=2, transform=transform)
                moved_cts.append(moved_ct)
                moved_labels.append(moved_label)
            moved_cts = np.stack(moved_cts, axis=0)
            moved_labels = np.stack(moved_labels, axis=0)

            logger.info(f'Saving corrected projected ct/labels for PAT{p} Fx{f:02d} Arc{a:02d}')
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\ct_proj_interp_corrected.npz"
            save_numpy(moved_cts, filepath)
            label_name = 'labels_proj_interp_corrected.npz' if labels_output == 'boolean' else 'labels_proj_interp_corrected_raw.npz'
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\{label_name}"
            save_numpy(moved_labels, filepath)
