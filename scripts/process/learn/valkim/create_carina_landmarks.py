import dicomset as ds
from dicomset.nifti.utils import create_landmarks, append_landmarks, landmarks_exist
import numpy as np
import pandas as pd
from collections import deque
from itertools import product
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

def get_trachea_split(mask: np.ndarray, affine: np.ndarray, k: int = 3):
    """Returns (trachea split in world coords, airway radius there in mm)."""
    mask = mask.astype(bool)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    radius = distance_transform_edt(mask, sampling=spacing)
    skel = skeletonize(mask) > 0
    skel_p = np.pad(skel, k)

    idx = np.indices((2*k+1,)*3) - k
    shell = np.abs(idx).max(axis=0) == k
    struct = np.ones((3, 3, 3))

    def is_junction(p):
        z, y, x = (c + k for c in p)
        patch = skel_p[z-k:z+k+1, y-k:y+k+1, x-k:x+k+1] & shell
        return label(patch, structure=struct)[1] >= 3

    pts = {tuple(p) for p in np.argwhere(skel)}
    offsets = [o for o in product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    nbrs = lambda p: [q for o in offsets
                      if (q := (p[0]+o[0], p[1]+o[1], p[2]+o[2])) in pts]

    seed = max(pts, key=lambda p: radius[p])
    seen, queue = {seed}, deque([seed])
    while queue:
        p = queue.popleft()
        if is_junction(p):
            world = (affine @ np.append(p, 1.0))[:3]
            return world, float(radius[p])
        for q in nbrs(p):
            if q not in seen:
                seen.add(q)
                queue.append(q)
    raise ValueError("No bifurcation found.")

def get_carina_point(mask: np.ndarray, affine: np.ndarray,
                     split_world, split_radius: float,
                     alpha: float = 3.0, min_area: int = 5):
    """Returns carina in world coords."""
    mask = mask.astype(bool)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    v = np.rint((np.linalg.inv(affine) @ np.append(split_world, 1.0))[:3]).astype(int)

    si = int(np.argmax(np.abs(affine[2, :3])))
    step = -int(np.sign(affine[2, si]))
    a, b = [ax for ax in range(3) if ax != si]
    zs = range(v[si], mask.shape[si] if step > 0 else -1, step)

    # central disk of radius alpha * r around the split
    ia, ib = np.meshgrid(np.arange(mask.shape[a]),
                         np.arange(mask.shape[b]), indexing="ij")
    disk = (((ia - v[a]) * spacing[a]) ** 2 +
            ((ib - v[b]) * spacing[b]) ** 2) <= (alpha * split_radius) ** 2

    for z in zs:
        lab, _ = label(np.take(mask, z, axis=si) & disk)
        areas = np.bincount(lab.ravel())[1:]
        big = [i + 1 for i in np.argsort(areas)[::-1][:2] if areas[i] >= min_area]
        if len(big) == 2:
            pa = np.argwhere(lab == big[0])
            pb = np.argwhere(lab == big[1])
            dist, i = cKDTree(pa * spacing[[a, b]]).query(pb * spacing[[a, b]])
            j = np.argmin(dist)
            mid = (pa[i[j]] + pb[j]) / 2.0

            vox = np.zeros(3)
            vox[a], vox[b], vox[si] = mid[0], mid[1], z - step  # slice above
            return (affine @ np.append(vox, 1.0))[:3]

    raise ValueError("Mask never splits into two regions below the trachea split.")

if __name__ == '__main__':
    dataset = 'VALKIM-PP'
    pat_ids = ['PAT1']
    study_id = 'study_0'
    series_ids = ['series_0']

    for p in tqdm(pat_ids, desc='Patients'):
        # Load data.
        dset = ds.load(dataset, 'nifti')
        pat = dset.patient(p)
        study = pat.study(study_id)
        for s in tqdm(series_ids, desc='Series', leave=False):
            ct_series = study.ct_series(s)
            ct, affine = ct_series.data, ct_series.affine
            rseries = study.regions_series(s)
            trachea_label = rseries.data(r='ts_trachea', rr=False)

            # Find where the trachea skeleton first branches into the two main bronchi.
            split_world, split_radius = get_trachea_split(trachea_label, affine)

            # Find the carina point itself - the ridge between the two bronchi, just below the split.
            carina_world = get_carina_point(trachea_label, affine, split_world, split_radius)

            landmark_ids = ['ts_carina']
            landmarks_data = np.stack([carina_world], axis=0)

            # Save landmarks to disk.
            df = pd.DataFrame([landmark_ids, landmarks_data[:, 0], landmarks_data[:, 1], landmarks_data[:, 2]]).T
            df = df.rename(columns={ 0: 'landmark-id', 1: 0, 2: 1, 3: 2 })
            if landmarks_exist(dataset, p, study_id, s):
                append_landmarks(dataset, p, study_id, s, df)
            else:
                create_landmarks(dataset, p, study_id, s, df)
