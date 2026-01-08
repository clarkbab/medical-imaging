from tqdm import tqdm

from mymi import datasets as ds
from mymi.transforms import sample
from mymi.utils import *

# Load patient.
dataset = 'PMCC-REIRRAD'
models = [
    'rir',
    'dmp',
    'edmp',
    'sg_c',
    'sg_lm',
]
set = ds.get(dataset, 'nifti')
dset = ds.get(dataset, 'dicom')
pat_ids = set.list_patients(group='hn')
dose_dfs = []
for p in tqdm(pat_ids):
    print(p)
    for m in models:
        print(m)
        # Get model/dose series.
        pat = set.patient(p)
        fixed_study = pat.study('idx:1')
        print(fixed_study)
        model_series = {}
        for s in fixed_study.list_dose_series():
            print(s)
            series = fixed_study.dose_series(s)
            if f'C2_PROP/RTDOSE/{m}.dcm' in series.dicom.filepath:
                model_series[m] = s

        # Sample moving dose at moving landmarks and moved dose at fixed landmarks - calculate difference.
        moving_study = pat.study('idx:0')
        moving_lm_series = moving_study.landmarks_series('series_1')
        assert '/C1/' in moving_lm_series.dicom.filepath, moving_lm_series.dicom.filepath
        moving_lms = moving_lm_series.data()
        moving_dose_series = moving_study.dose_series('series_2')
        assert '/C1/' in moving_dose_series.dicom.filepath, moving_dose_series.dicom.filepath
        moving_dose = moving_dose_series.data
        moving_df = sample(moving_dose, moving_lms, spacing=moving_dose_series.spacing, origin=moving_dose_series.origin)
        
        fixed_study = pat.study('idx:1')
        fixed_lm_series = fixed_study.landmarks_series('series_1')
        assert '/C2/' in fixed_lm_series.dicom.filepath, fixed_lm_series.dicom.filepath
        fixed_lms = fixed_lm_series.data()
        moved_dose_series = fixed_study.dose_series(model_series[m])
        assert f'/C2_PROP/RTDOSE/{m}.dcm' in moved_dose_series.dicom.filepath, moved_dose_series.dicom.filepath
        moved_dose = moved_dose_series.data
        moved_df = sample(moved_dose, fixed_lms, spacing=moved_dose_series.spacing, origin=moved_dose_series.origin)
        
        moved_df['diff'] = moved_df['sample'] - moving_df['sample']
        moved_df['diff-abs'] = moved_df['diff'].abs()
        moved_df.insert(3, 'model', m)
        dose_dfs.append(moved_df)

dose_df = pd.concat(dose_dfs, axis=0)
dose_df.insert(4, 'metric', 'dose-error')
dose_df = dose_df.rename(columns={ 'diff-abs': 'value' })

save_csv(dose_df, 'files:imreg/hn-dose.csv')
