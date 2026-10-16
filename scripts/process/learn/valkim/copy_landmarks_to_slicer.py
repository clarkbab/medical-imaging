import dicomset as ds
from dicomset.typing import *
from dicomset.utils import resolve_filepath, landmarks_to_points, save_nifti, change_image_orientation
import numpy as np
import os
from tqdm.auto import tqdm

def write_fcsv(
    landmarks: Landmarks3D,
    filepath: FilePath,
    coords: Literal['LSP', 'RAS'] = 'LPS',
    ) -> None:
    filepath = resolve_filepath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    labels = landmarks['landmark-id'].tolist()
    points = landmarks_to_points(landmarks)
    n = points.shape[0]
    with open(filepath, "w") as f:
        f.write("# Markups fiducial file version = 5.3\n")
        f.write(f"# CoordinateSystem = {coords}\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for l, (x, y, z) in zip(labels, points):
            f.write(f"{l},{x},{y},{z},0,0,0,1,1,1,0,,{l},\n")

if __name__ == '__main__':
    dataset = 'VALKIM-PP'
    pat_ids = ['PAT1']
    study_id = 'study_0'
    series_ids = ['series_0']

    for p in tqdm(pat_ids):
        # Load data.
        dset = ds.load(dataset, 'nifti')
        pat = dset.patient(p)
        study = pat.study(study_id)
        for s in tqdm(series_ids, leave=False):
            ctseries = study.ct_series(s)
            ct, affine = ctseries.data, ctseries.affine
            rseries = study.regions_series(s)
            rlabels, rdata = rseries.data(r=[
                'ts_costal_cartilages',
                'ts_Lung',
                'ts_lung_airways',
                'ts_Ribs',
                'ts_sternum',
                'ts_Vertebrae_Body_T',
                'ts_Vertebrae_T',
            ], rr=True)
            lseries = study.landmarks_series(s)
            ldata = lseries.data()

            # Save CT.
            f = f'files:learn/valkim/landmarks/{p}/{s}/ct.nii.gz'
            ct_ras, affine_ras = change_image_orientation(ct, 'LPS', 'RAS', a=affine)
            if not os.path.exists(f):
                save_nifti(ct_ras, affine_ras, f)

            # Save the labels that were used to derive the landmarks.
            f = f'files:learn/valkim/{p}/{s}/ts_labels.nii.gz'
            rdata_ras, _ = change_image_orientation(rdata, 'LPS', 'RAS', a=affine)
            rdata_slicer = np.concatenate([np.zeros(rdata_ras.shape[1:])[np.newaxis, ...], rdata_ras], axis=0)
            rdata_slicer = np.argmax(rdata_slicer, axis=0).astype(np.int16)
            if not os.path.exists(f):
                save_nifti(rdata_slicer, affine_ras, f)

            # Save all landmarks.
            f = f'files:learn/valkim/{p}/{s}/landmarks.fcsv'
            write_fcsv(ldata, f)
