import dicomset as ds
from dicomset.nifti.utils import create_landmarks
from dicomset.utils import centre_of_mass, to_numpy
import pandas as pd
from tqdm.auto import tqdm

dataset = 'VALKIM-PP'
pats = ['PAT1']
study_id = 'study_0'
series = ['series_0']
series = ['series_1', 'series_2', 'series_3', 'series_4', 'series_5',
          'series_6', 'series_7', 'series_8', 'series_9']
series = ['series_8', 'series_9']

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
        rlabels, rdata = rseries.data(r='ts_Vertebrae_Body_T')

        # Get centroids.
        # Extract centroid for each vertebra.
        print(rdata.shape)
        centroids = centre_of_mass(rdata, affine=affine)
        centroids = to_numpy(centroids)

        # Save landmarks to disk.
        df = pd.DataFrame([rlabels, centroids[:, 0], centroids[:, 1], centroids[:, 2]]).T
        df = df.rename(columns={ 0: 'landmark-id', 1: 0, 2: 1, 3: 2 })
        create_landmarks(dataset, p, study_id, s, df)
