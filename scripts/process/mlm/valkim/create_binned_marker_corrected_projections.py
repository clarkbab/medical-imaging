from dicomset.utils import load_numpy, save_numpy, to_numpy, centre_of_mass, resample, logger
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm

from mymi.utils.cdog import load_tiff_arc

pat_ids = [1]
fractions = [1]

# We only have manual marker segmentation for the first arc of each fraction.
arcs = [1]

for p in tqdm(pat_ids):
    for f in tqdm(fractions, leave=False):
        for a in tqdm(arcs, leave=False):
            assert a == 1, "Only first arc has manual marker segmentation."

            # Load IF markers.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient{p:02d}\Fx{f:02d}\MarkerLocationGA_CouchShift_adapted.txt"
            df = pd.read_csv(filepath)
            if_markers = df[['Marker_0_X', 'Marker_0_Y', 'Marker_1_X', 'Marker_1_Y', 'Marker_2_X', 'Marker_2_Y']].to_numpy()
            if_markers = if_markers.reshape((-1, 3, 2))

            # Load IF images.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}\Fx{f:02d}"
            frames, info = load_tiff_arc(filepath, arc=a)

            # Load projected CT.
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\ct_proj_binned.npz"
            ct = load_numpy(filepath)

            # Load projected labels.
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\labels_proj_binned.npz"
            labels = load_numpy(filepath)
            gtv = labels[:, 0]

            # Flip markers.
            if_markers[:, :, 1] = frames.shape[2] - if_markers[:, :, 1]

            # Plot.
            moved_cts = []
            moved_labels = []
            for frame in tqdm(range(len(frames))):
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
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\ct_proj_binned_corrected.npz"
            save_numpy(moved_cts, filepath)
            filepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\labels_proj_binned_corrected.npz"
            save_numpy(moved_labels, filepath)
