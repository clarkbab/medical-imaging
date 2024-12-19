import os
import shutil

from mymi import dataset as ds

if True:
    set = ds.get('MICCAI-2015', 'nrrd')
    print(set.list_patients())

    pat_id = set.list_patients()[0]
    pat = set.patient(pat_id)

    print(pat.list_studies())

    study_id = pat.list_studies()[0]
    study = pat.study(study_id)

    print(study.list_data('CT'))
    print(study.list_data('REGIONS'))
    
    ct_id = study.list_data('CT')[0]
    regions_id = study.list_data('REGIONS')[0]
    ct_data = study.data(ct_id, 'CT').data
    regions_data = study.data(regions_id, 'REGIONS').data()
    
    print(ct_data.shape)
    print(len(regions_data.keys()))
    print(regions_data[list(regions_data.keys())[0]].shape)

if False:
    basepath = '/data/gpfs/projects/punim1413/mymi/datasets/nrrd/MICCAI-2015/data/patients' 

    files = os.listdir(os.path.join(basepath, 'ct'))
    files = [f.split('.')[0] for f in files]

    print(files)

    regions = list(sorted(os.listdir(os.path.join(basepath, 'regions'))))

    print(regions)

    for f in files:
        for r in regions:
            src = os.path.join(basepath, 'regions', r, f'{f}.nrrd')
            if os.path.exists(src):
                dest = os.path.join(basepath, f, 'study_0', 'regions', 'series_1', f'{f}.nrrd')
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copyfile(src, dest)
