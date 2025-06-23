import os
import torch
from tqdm import tqdm

from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate
from mymi.utils import *

from ...processing import fill_border_padding, fill_contiguous_padding

def convert_lung250m_4b_to_nifti() -> None:
    dry_run = False
    rdataset = 'Lung250M-4B'
    dataset = 'LUNG250M-4B'
    replace_padding = True
    fill = -2000
    rset = RawDataset(rdataset)
    set = recreate(dataset)
    from mymi.datasets import NiftiDataset
    set = NiftiDataset(dataset)

    # Create holdout split file.
    filepath = os.path.join(rset.path, 'imagesTr')
    train_ids = list(sorted(np.unique(['_'.join(i.split('_')[:2]) for i in os.listdir(filepath)])))
    train_df = pd.DataFrame(np.transpose([train_ids, ['train'] * len(train_ids)]), columns=['patient-id', 'split'])
    filepath = os.path.join(rset.path, 'imagesTs')

    # Contains both validation and test IDs.
    all_test_ids = list(sorted(np.unique(['_'.join(i.split('_')[:2]) for i in os.listdir(filepath)])))
    val_ids = [i for i in all_test_ids if int(i.split('_')[1]) >= 104 and int(i.split('_')[1]) <= 113]
    test_ids = [i for i in all_test_ids if i not in val_ids]
    val_df = pd.DataFrame(np.transpose([val_ids, ['validate'] * len(val_ids)]), columns=['patient-id', 'split'])
    test_df = pd.DataFrame(np.transpose([test_ids, ['test'] * len(test_ids)]), columns=['patient-id', 'split'])
    df = pd.concat((train_df, val_df, test_df), axis=0)
    filepath = os.path.join(set.path, 'holdout-split.csv')
    save_csv(df, filepath)

    # Load manual landmarks.
    filepath = os.path.join(rset.path, 'lms_validation.pth')
    val_lm_data = torch.load(filepath)

    # Copy patient data.
    fixed_suffix = 1
    fixed_study = 'study_1'
    moving_suffix = 2
    moving_study = 'study_0'
    for p in tqdm(df['patient-id']):
        split = 'Tr' if len(df[(df['patient-id'] == p) & (df['split'] == 'train')]) > 0 else 'Ts'
        
        # Copy fixed image data.
        filepath = os.path.join(rset.path, f'images{split}', f'{p}_{fixed_suffix}.nii.gz')
        ct_data, fixed_spacing, offset = load_nifti(filepath)
        if replace_padding:
            ct_data = fill_border_padding(ct_data, fill=fill)
            # ct_data = fill_contiguous_padding(ct_data, fill=fill)
        fixed_spacing = tuple(np.abs(fixed_spacing))    # Spacing was saved as (-1, -1, 1), presumably because their code used LPS coordinates.
        offset = (0, 0, 0)      # Set this to zero as it doesn't really matter.
        filepath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'ct', 'series_0.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_nifti(ct_data, filepath, spacing=fixed_spacing, offset=offset)

        # Copy moving image data.
        filepath = os.path.join(rset.path, f'images{split}', f'{p}_{moving_suffix}.nii.gz')
        ct_data, moving_spacing, offset = load_nifti(filepath)
        if replace_padding:
            ct_data = fill_border_padding(ct_data, fill=fill)
            # ct_data = fill_contiguous_padding(ct_data, fill=fill)
        moving_spacing = tuple(np.abs(moving_spacing))    # Spacing was saved as (-1, -1, 1), presumably because their code used LPS coordinates.
        offset = (0, 0, 0)      # Set this to zero as it doesn't really matter.
        filepath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'ct', 'series_0.nii.gz')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_nifti(ct_data, filepath, spacing=moving_spacing, offset=offset)

        # Copy lung masks.
        filepath = os.path.join(rset.path, f'masks{split}', f'{p}_{fixed_suffix}.nii.gz')
        lung_data, lung_spacing, lung_offset = load_nifti(filepath)
        destpath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(lung_data, destpath, spacing=lung_spacing, offset=lung_offset)

        filepath = os.path.join(rset.path, f'masks{split}', f'{p}_{moving_suffix}.nii.gz')
        lung_data, lung_spacing, lung_offset = load_nifti(filepath)
        destpath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(lung_data, destpath, spacing=lung_spacing, offset=lung_offset)

        # Copy artery/vein segmentations - guessing arteries are first...
        filepath = os.path.join(rset.path, f'seg{split}', f'{p}_{fixed_suffix}.nii.gz')
        seg_data, seg_spacing, seg_offset = load_nifti(filepath)
        artery_data = seg_data == 1
        destpath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'regions', 'series_1', 'Arteries.nii.gz')
        save_nifti(artery_data, destpath, spacing=seg_spacing, offset=seg_offset)
        vein_data = seg_data == 2
        destpath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'regions', 'series_1', 'Veins.nii.gz')
        save_nifti(vein_data, destpath, spacing=seg_spacing, offset=seg_offset)

        filepath = os.path.join(rset.path, f'seg{split}', f'{p}_{moving_suffix}.nii.gz')
        seg_data, seg_spacing, seg_offset = load_nifti(filepath)
        artery_data = seg_data == 1
        destpath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'regions', 'series_1', 'Arteries.nii.gz')
        save_nifti(artery_data, destpath, spacing=seg_spacing, offset=seg_offset)
        vein_data = seg_data == 2
        destpath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'regions', 'series_1', 'Veins.nii.gz')
        save_nifti(vein_data, destpath, spacing=seg_spacing, offset=seg_offset)

        # Copy landmarks.
        if split == 'Tr':
            # Load points.
            filepath = os.path.join(rset.path, f'corrfield{split}', f'{p}.csv')
            points = pd.read_csv(filepath, header=None)

            # Split landmarks.
            fixed_points = points[list(range(3))].copy()
            fixed_points.insert(0, 'landmark-type', 'corrfield')
            moving_points = points[list(range(3, 6))].copy().rename(columns=dict((i + 3, i) for i in range(3)))
            moving_points.insert(0, 'landmark-type', 'corrfield')
        elif split == 'Ts':
            # Load points.
            points = val_lm_data[str(int(p.split('_')[1]))]

            # Split landmarks.
            fixed_points = pd.DataFrame(points[:, :3])
            fixed_points.insert(0, 'landmark-type', 'manual')
            moving_points = pd.DataFrame(points[:, 3:])
            moving_points.insert(0, 'landmark-type', 'manual')
            
        # Convert landmarks from image coordinates to patient coordinates.
        fixed_points_data = fixed_points[list(range(3))].to_numpy()
        fixed_points_data = fixed_spacing * fixed_points_data
        fixed_points[list(range(3))] = fixed_points_data
        fixed_points.insert(0, 'landmark-id', list(range(len(fixed_points))))
        moving_points_data = moving_points[list(range(3))].to_numpy()
        moving_points_data = moving_spacing * moving_points_data
        moving_points[list(range(3))] = moving_points_data
        moving_points.insert(0, 'landmark-id', list(range(len(moving_points))))

        if not dry_run:
            filepath = os.path.join(set.path, 'data', 'patients', p, fixed_study, 'landmarks', 'series_1.csv')
            save_csv(fixed_points, filepath, header=True, index=False)
            filepath = os.path.join(set.path, 'data', 'patients', p, moving_study, 'landmarks', 'series_1.csv')
            save_csv(moving_points, filepath, header=True, index=False)
