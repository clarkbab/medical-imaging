import dicomset as ds
from dicomset.nifti.utils import create_landmarks, landmarks_exist, append_landmarks
from dicomset.utils import to_world_coords
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import networkx as nx

def label_to_graph(mask: np.ndarray) -> nx.Graph:
    coords = np.argwhere(mask)
    coord_set = {tuple(c) for c in coords}
    G = nx.Graph()
    offsets = [(dx-1, dy-1, dz-1) for dx,dy,dz in np.ndindex(3,3,3) if (dx,dy,dz)!=(1,1,1)]

    for c in coords:
        c = tuple(c)
        G.add_node(c)
        for o in offsets:
            n = tuple(np.add(c, o))
            if n in coord_set:
                dist = np.linalg.norm(o)  # $\sqrt{dx^2+dy^2+dz^2}$
                G.add_edge(c, n, weight=dist)
    return G

# Starting from a random node, we find the furthest node by counting
# edge weights - this is one end of the rib.
def farthest_node(graph, start):
    lengths = nx.single_source_dijkstra_path_length(graph, start, weight="weight")
    return max(lengths, key=lengths.get), lengths

def sternum_landmarks(mask: np.ndarray):
    graph = label_to_graph(mask)
    coords = np.argwhere(mask)
    # Start from superior point so that we find lower end of rib first.
    start = tuple(coords[np.argmax(coords[:, 2])])
    # Find lower end of the rib.
    low, _ = farthest_node(graph, start)
    # Find the upper end of the rib.
    high, dists_from_low = farthest_node(graph, low)
    # Find the middle of the rib.
    total_dist = dists_from_low[high]
    path = nx.shortest_path(graph, low, high, weight='weight')
    total = 0.0
    mid = path[0]
    for a, b in zip(path[:-1], path[1:]):
        total += graph[a][b]['weight']
        mid = b
        if total >= total_dist / 2:
            break

    return np.stack([low, mid, high], axis=0)

if __name__ == '__main__':
    dataset = 'VALKIM-PP'
    pats = ['PAT1']
    study_id = 'study_0'
    series = ['series_0']
    # series = ['series_1', 'series_2', 'series_3', 'series_4', 'series_5',
    #           'series_6', 'series_7', 'series_8', 'series_9']

    set = ds.load(dataset, 'nifti')

    for p in tqdm(pats, desc='Patients'):
        print(p)
        for s in tqdm(series, desc='Series', leave=False):
            print(s)
            # Load data.
            pat = set.patient(p)
            study = pat.study(study_id)
            series = study.ct_series(s)
            ct, affine = series.data, series.affine
            rseries = study.regions_series(s)
            rlabels, rdata = rseries.data(r='ts_sternum')

            # Get landmarks.
            landmark_ids = []
            landmarks_data = []

            # Get rib landmarks.
            d_lms = sternum_landmarks(rdata)
            d_lms = to_world_coords(d_lms, affine)

            landmark_ids.append(f'ts_sternum_low')
            landmarks_data.append(d_lms[0])
            landmark_ids.append(f'ts_sternum_mid')
            landmarks_data.append(d_lms[1])
            landmark_ids.append(f'ts_sternum_high')
            landmarks_data.append(d_lms[2])

            landmarks_data = np.stack(landmarks_data, axis=0)

            # Save landmarks to disk.
            df = pd.DataFrame([landmark_ids, landmarks_data[:, 0], landmarks_data[:, 1], landmarks_data[:, 2]]).T
            df = df.rename(columns={ 0: 'landmark-id', 1: 0, 2: 1, 3: 2 })
            if landmarks_exist(dataset, p, study_id, s):
                append_landmarks(dataset, p, study_id, s, df)
            else:
                create_landmarks(dataset, p, study_id, s, df)
