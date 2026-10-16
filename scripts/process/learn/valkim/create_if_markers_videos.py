import pandas as pd
from tqdm.auto import tqdm

from mymi.utils.cdog import load_tiff_arc
from mymi.utils.plotting import play_video
from mymi.utils.projections import plot_projection

pat_ids = [1]
fractions = [1]
arcs = [1, 2, 3]
n_frames = None

for p in tqdm(pat_ids):
    for f in tqdm(fractions, leave=False):
        for a in tqdm(arcs, leave=False):
            # Load IF markers.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient{p:02d}\Fx{f:02d}\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
            df = pd.read_csv(filepath)
            markers = df[['Marker_0_X', 'Marker_0_Y', 'Marker_1_X', 'Marker_1_Y', 'Marker_2_X', 'Marker_2_Y']].to_numpy()
            markers = markers.reshape((-1, 3, 2))

            # Load IF images.
            filepath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}\Fx{f:02d}"
            arc_data, arc_info = load_tiff_arc(filepath, arc=a)

            # Flip markers.
            markers[:, :, 1] = arc_data.shape[2] - markers[:, :, 1]

            plot_args = [(arc_data[i], arc_info.iloc[i]) for i in range(len(arc_info))]
            plot_kwargs = [{'all_info': arc_info, 'points': markers[i]} for i in range(len(arc_info))]

            savepath = rf"D:\Brett\data\mymi\datasets\nifti\VALKIM-PP\data\projections\PAT{p}\Fx{f:02d}\Arc{a:02d}\if_markers.mp4"
            play_video(plot_projection, plot_args, plot_kwargs, n_frames=n_frames, savepath=savepath, show=False, overwrite=True)
