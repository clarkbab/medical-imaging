from dicomset.utils import save_numpy
from mymi.utils.cdog import load_tiff_arc
import pandas as pd

arcs = [1, 2, 3]
ff = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx01"
for a in arcs:
    data, info = load_tiff_arc(ff, a)
    
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient01\Fx01\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
    df = pd.read_csv(f)
    points = df[['Marker_0_X', 'Marker_0_Y', 'Marker_1_X', 'Marker_1_Y', 'Marker_2_X', 'Marker_2_Y']].to_numpy()
    points = points.reshape(-1, 3, 2)
    points[:, :, 1] = data[0].shape[1] - points[:, :, 1]  # Y-coord needs to be flipped.

    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient01\Fx01\Arc{a:02d}\markers.npz"
    save_numpy(points, f)
