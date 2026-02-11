import os

from mymi import datasets as ds
from mymi.datasets.dicom import DicomStudy

def study_sort(s: DicomStudy) -> int:
    assert s.has_rtstruct
    filepath = s.rtstruct_series('i:0').filepath
    if '/C1/' in filepath:
        return 0
    elif '/C2/' in filepath:
        return 1
    raise ValueError(f"Study is not C1 or C2. Filepath: {filepath}")

# Generate marker files.
dataset = 'PMCC-REIRRAD'
dset = ds.get(dataset, 'dicom')
pat_ids = dset.list_patients(group='lung')

for p in pat_ids:
    pat = dset.patient(p)
    moving_study = pat.study('i:0', sort=study_sort)
    fixed_study = pat.study('i:1', sort=study_sort)
    moving_series = moving_study.rtstruct_series('i:0')
    assert '/C1/' in moving_series.filepath, moving_series.filepath
    moving_lms = moving_series.landmarks_data()
    fixed_series = fixed_study.rtstruct_series('i:0')
    assert '/C2/' in fixed_series.filepath
    fixed_lms = fixed_series.landmarks_data()
    filepath = os.path.join(dset.path, 'data', 'velocity', pat.id, 'C1-markers.txt')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        n_lines = len(moving_lms)
        for i in range(n_lines):
            line = ' '.join([str(float(v)) for v in moving_lms.iloc[i][[0, 1, 2]].tolist()]) + '\n'
            f.write(line)
    filepath = os.path.join(dset.path, 'data', 'velocity', pat.id, 'C2-markers.txt')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        n_lines = len(fixed_lms)
        for i in range(n_lines):
            line = ' '.join([str(float(v)) for v in fixed_lms.iloc[i][[0, 1, 2]].tolist()]) + '\n'
            f.write(line)
