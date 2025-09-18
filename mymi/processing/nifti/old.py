import os
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.utils import *

def convert_old_pmcc_hn() -> None:
    datasets = ['PMCC-HN-TRAIN-OLD', 'PMCC-HN-TEST-OLD']
    for d in datasets:
        oset = NiftiDataset(d)
        filepath = os.path.join(oset.path, 'data', 'ct')
        files = os.listdir(filepath)
        pat_ids = list(np.unique([f.split('-')[0] for f in files]))

        # Get region names.
        filepath = os.path.join(oset.path, 'data', 'regions')
        regions = os.listdir(filepath)

        # Create dataset.
        from mymi.datasets.nifti import recreate
        nset = recreate(d.replace('-OLD', ''))

        fixed_study = 'study_1'
        moving_study = 'study_0'
        fixed_id = 1
        moving_id = 0
        for p in tqdm(pat_ids):
            # Copy CT data.
            filepath = os.path.join(oset.path, 'data', 'ct', f'{p}-{fixed_id}.nii.gz')
            data, spacing, origin = load_nifti(filepath)
            filepath = os.path.join(nset.path, 'data', 'patients', p, fixed_study, 'ct', 'series_0.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_nifti(data, filepath, spacing=spacing, origin=origin)
            
            filepath = os.path.join(oset.path, 'data', 'ct', f'{p}-{moving_id}.nii.gz')
            data, spacing, origin = load_nifti(filepath)
            filepath = os.path.join(nset.path, 'data', 'patients', p, moving_study, 'ct', 'series_0.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_nifti(data, filepath, spacing=spacing, origin=origin)
            
            # Copy region data.
            for r in regions:
                filepath = os.path.join(oset.path, 'data', 'regions', r, f'{p}-{fixed_id}.nii.gz')
                if os.path.exists(filepath):
                    data, spacing, origin = load_nifti(filepath)
                    filepath = os.path.join(nset.path, 'data', 'patients', p, fixed_study, 'regions', 'series_1', f'{r}.nii.gz')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    save_nifti(data, filepath, spacing=spacing, origin=origin)
                    
                filepath = os.path.join(oset.path, 'data', 'regions', r, f'{p}-{moving_id}.nii.gz')
                if os.path.exists(filepath):
                    data, spacing, origin = load_nifti(filepath)
                    filepath = os.path.join(nset.path, 'data', 'patients', p, moving_study, 'regions', 'series_1', f'{r}.nii.gz')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    save_nifti(data, filepath, spacing=spacing, origin=origin)

def convert_old_replan() -> None:
    oset = NiftiDataset('PMCC-HN-REPLAN-OLD')
    filepath = os.path.join(oset.path, 'data', 'ct')
    files = os.listdir(filepath)
    pat_ids = list(np.unique([f.split('-')[0] for f in files]))

    # Get region names.
    filepath = os.path.join(oset.path, 'data', 'regions')
    regions = os.listdir(filepath)

    # Create dataset.
    from mymi.datasets.nifti import recreate
    nset = recreate('PMCC-HN-REPLAN')

    fixed_study = 'study_1'
    moving_study = 'study_0'
    fixed_id = 1
    moving_id = 0
    for p in tqdm(pat_ids):
        # Copy CT data.
        filepath = os.path.join(oset.path, 'data', 'ct', f'{p}-{fixed_id}.nii.gz')
        data, spacing, origin = load_nifti(filepath)
        filepath = os.path.join(nset.path, 'data', 'patients', p, fixed_study, 'ct', 'series_0.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_nifti(data, filepath, spacing=spacing, origin=origin)
        
        filepath = os.path.join(oset.path, 'data', 'ct', f'{p}-{moving_id}.nii.gz')
        data, spacing, origin = load_nifti(filepath)
        filepath = os.path.join(nset.path, 'data', 'patients', p, moving_study, 'ct', 'series_0.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_nifti(data, filepath, spacing=spacing, origin=origin)
        
        # Copy region data.
        for r in regions:
            filepath = os.path.join(oset.path, 'data', 'regions', r, f'{p}-{fixed_id}.nii.gz')
            if os.path.exists(filepath):
                data, spacing, origin = load_nifti(filepath)
                filepath = os.path.join(nset.path, 'data', 'patients', p, fixed_study, 'regions', 'series_1', f'{r}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                save_nifti(data, filepath, spacing=spacing, origin=origin)
                
            filepath = os.path.join(oset.path, 'data', 'regions', r, f'{p}-{moving_id}.nii.gz')
            if os.path.exists(filepath):
                data, spacing, origin = load_nifti(filepath)
                filepath = os.path.join(nset.path, 'data', 'patients', p, moving_study, 'regions', 'series_1', f'{r}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                save_nifti(data, filepath, spacing=spacing, origin=origin)
