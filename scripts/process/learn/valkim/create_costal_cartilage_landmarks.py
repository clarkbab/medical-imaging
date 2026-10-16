import dicomset as ds
from dicomset.utils import to_image_coords, to_world_coords, affine_spacing
from dicomset.nifti.utils import create_landmarks, append_landmarks, landmarks_exist
import networkx as nx
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from tqdm.auto import tqdm

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

def closest_node(graph, seed_point):
    nodes = np.array(graph.nodes())
    dists = np.linalg.norm(nodes - np.array(seed_point), axis=1)
    return tuple(nodes[np.argmin(dists)])

def break_spindles(mask, affine, thresh=2.0, cut_radius_mm=3.0):
    spacing = affine_spacing(affine)
    dist = distance_transform_edt(mask, sampling=spacing)
    skel = skeletonize(mask)
    skel_pts = np.argwhere(skel)

    G = nx.Graph()
    pts_set = set(map(tuple, skel_pts))
    for p in skel_pts:
        G.add_node(tuple(p), dist=dist[tuple(p)])
        for d in np.ndindex(3, 3, 3):
            n = tuple(p + np.array(d) - 1)
            if n != tuple(p) and n in pts_set:
                G.add_edge(tuple(p), n)

    necks = [n for n in G.nodes if G.nodes[n]['dist'] < thresh
             and all(G.nodes[n]['dist'] <= G.nodes[nb]['dist'] for nb in G.neighbors(n))]

    # Convert mm radius to voxel radius per axis (handles anisotropic spacing)
    cut_radius_vox = np.ceil(cut_radius_mm / np.array(spacing)).astype(int)

    broken_mask = mask.copy()
    for n in necks:
        zc, yc, xc = n
        rz, ry, rx = cut_radius_vox
        sl = (
            slice(max(zc - rz, 0), zc + rz + 1),
            slice(max(yc - ry, 0), yc + ry + 1),
            slice(max(xc - rx, 0), xc + rx + 1),
        )
        broken_mask[sl] = False

    labeled, n_components = label(broken_mask)
    print(f"Found {n_components} components after breaking necks")
    return broken_mask, labeled

if __name__ == '__main__':
    dataset = 'VALKIM-PP'
    pat_ids = ['PAT1']
    study_id = 'study_0'
    series_ids = ['series_0']

    # Can only obtain ribs 1-5 landmarks easily.
    ribs = list(range(1, 6))
    sides = ['left', 'right']

    for p in tqdm(pat_ids):
        # Load data.
        dset = ds.load(dataset, 'nifti')
        pat = dset.patient(p)
        study = pat.study(study_id)
        for s in tqdm(series_ids, leave=False):
            ct_series = study.ct_series(s)
            ct, affine = ct_series.data, ct_series.affine
            rseries = study.regions_series(s)
            lseries = study.landmarks_series(s)
            cc_label = rseries.data(r='ts_costal_cartilages', rr=False)
            sternum_label = rseries.data(r='ts_sternum', rr=False)
            ldata = lseries.data()

            # Break spindle connections between cartilage segments - otherwise
            # these confuse the furthest point calculation.
            cc_broken, _ = break_spindles(cc_label, affine)

            # Build graph of remaining cartilage mask.
            graph = label_to_graph(cc_broken)

            # Get distances from sternum, useful for getting point on 
            # cartilage segment that is closest to the sternum.
            sternum_dists = distance_transform_edt(~sternum_label)

            # For each rib/side, get the upper and lower landmarks.
            # Upper landmark is very close to lower rib landmarks, and probably
            # redundant for downstream tasks.
            landmark_ids = []
            landmarks_data = []
            for r in ribs:
                for sd in sides:
                    # Find a point in the costal cartilage that is closest to the lower rib landmark.
                    rib_low = ldata[ldata['landmark-id'] == f'ts_rib_{sd}_{r}_low'][[0, 1, 2]].iloc[0].values
                    rib_low_vox = to_image_coords(rib_low, affine)
                    costal_high_vox = closest_node(graph, rib_low_vox)
                    costal_high = to_world_coords(costal_high_vox, affine)
                    
                    # Find a connected point that is closest to the sternum.
                    costal_nodes = nx.node_connected_component(graph, costal_high_vox)
                    costal_low_vox = min(costal_nodes, key=lambda n: sternum_dists[tuple(np.round(n).astype(int))])
                    costal_low = to_world_coords(costal_low_vox, affine)

                    landmark_ids.append(f'ts_costal_{sd}_{r}_low')
                    landmark_ids.append(f'ts_costal_{sd}_{r}_high')
                    landmarks_data.append(costal_low)
                    landmarks_data.append(costal_high)
            landmarks_data = np.stack(landmarks_data, axis=0)

            # Save landmarks to disk.
            df = pd.DataFrame([landmark_ids, landmarks_data[:, 0], landmarks_data[:, 1], landmarks_data[:, 2]]).T
            df = df.rename(columns={ 0: 'landmark-id', 1: 0, 2: 1, 3: 2 })
            if landmarks_exist(dataset, p, study_id, s):
                append_landmarks(dataset, p, study_id, s, df)
            else:
                create_landmarks(dataset, p, study_id, s, df)
