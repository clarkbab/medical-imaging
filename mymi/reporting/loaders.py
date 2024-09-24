from fpdf import FPDF, TitleStyle
import numpy as np
import os
import pandas as pd
from pytorch_lightning import seed_everything
from tqdm import tqdm
from typing import List, Optional, Union
from uuid import uuid1

from mymi import config
from mymi.dataset.training import TrainingDataset 
from mymi.dataset.training_adaptive import TrainingAdaptiveDataset
from mymi.loaders import AdaptiveLoader, Loader, MultiLoader, MultiLoaderV2, RegSegLoader
from mymi.loaders.augmentation import get_transforms
from mymi import logging
from mymi.plotting import plot_patient
from mymi.types import PatientID, PatientRegions
from mymi.utils import append_row, arg_to_list, encode, load_csv, save_csv

def get_loader_manifest(
    dataset: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'region': str,
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': int,
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, TrainingDataset(d, check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    tl, vl, tsl = Loader.build_loaders(datasets, region, check_processed=check_processed, load_data=False, load_test_origin=False, shuffle_train=False, **kwargs)
    loader_names = ['train', 'validate', 'test']

    # Get values for this region.
    for loader_name, loader in zip(loader_names, (tl, vl, tsl)):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                origin_ds, origin_pat_id = dataset_map[dataset].sample(sample_id).origin
                data = {
                    'region': region,
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def create_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    logging.info(f"Creating loader manifest for datasets '{datasets}', region '{region}', n_folds '{n_folds}', test_fold '{test_fold}'.")

    # Get manifest.
    df = get_loader_manifest(datasets, region, check_processed=check_processed, n_folds=n_folds, test_fold=test_fold)

    # Save manifest.
    save_csv(df, 'loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv', index=False, overwrite=True)

def load_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    df = load_csv('loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv')
    df = df.astype({ 'origin-patient-id': str, 'sample-id': str })
    return df

def get_test_fold(
    datasets: Union[str, List[str]],
    dataset: str,
    pat_id: PatientID,
    region: str):
    for test_fold in range(5):
        df = load_loader_manifest(datasets, region, test_fold=test_fold)
        df = df[df.loader == 'test']
        df = df[(df['origin-dataset'] == dataset) & (df['origin-patient-id'] == str(pat_id))]
        if len(df) == 1:
            return test_fold

    raise ValueError(f"Patient '{pat_id}' not found for region '{region}' loader and dataset '{dataset}'.")

def get_reg_seg_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    use_split_file: bool = False,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': int,
        'group-id': float,      # Can contain 'nan' values.
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, TrainingAdaptiveDataset(d, check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    loaders = RegSegLoader.build_loaders(datasets, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_subfolds=n_subfolds, shuffle_train=False, use_split_file=use_split_file, **kwargs)
    if n_folds is not None or use_split_file:
        if n_subfolds is not None:
            loader_names = ['train', 'validate', 'subtest', 'test']
        else:
            loader_names = ['train', 'validate', 'test']
    else:
        loader_names = ['train', 'validate']

    # Get values for this region.
    for loader, loader_name in zip(loaders, loader_names):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                sample = dataset_map[dataset].sample(sample_id)
                group_id = sample.group_id
                origin_ds, origin_pat_id = sample.origin
                data = {
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'group-id': group_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def get_adaptive_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    use_split_file: bool = False,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': int,
        'group-id': float,      # Can contain 'nan' values.
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, TrainingAdaptiveDataset(d, check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    loaders = AdaptiveLoader.build_loaders(datasets, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_subfolds=n_subfolds, shuffle_train=False, use_split_file=use_split_file, **kwargs)
    if n_folds is not None or use_split_file:
        if n_subfolds is not None:
            loader_names = ['train', 'validate', 'subtest', 'test']
        else:
            loader_names = ['train', 'validate', 'test']
    else:
        loader_names = ['train', 'validate']

    # Get values for this region.
    for loader, loader_name in zip(loaders, loader_names):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                sample = dataset_map[dataset].sample(sample_id)
                group_id = sample.group_id
                origin_ds, origin_pat_id = sample.origin
                data = {
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'group-id': group_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def get_multi_loader_v2_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    use_split_file: bool = False,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': int,
        'group-id': float,      # Can contain 'nan' values.
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, TrainingAdaptiveDataset(d, check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    loaders = MultiLoaderV2.build_loaders(datasets, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_subfolds=n_subfolds, shuffle_train=False, use_split_file=use_split_file, **kwargs)
    if n_folds is not None or use_split_file:
        if n_subfolds is not None:
            loader_names = ['train', 'validate', 'subtest', 'test']
        else:
            loader_names = ['train', 'validate', 'test']
    else:
        loader_names = ['train', 'validate']

    # Get values for this region.
    for loader, loader_name in zip(loaders, loader_names):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                sample = dataset_map[dataset].sample(sample_id)
                group_id = sample.group_id
                origin_ds, origin_pat_id = sample.origin
                data = {
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'group-id': group_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def get_multi_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    use_split_file: bool = False,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': int,
        'input-shape': str,
        'group-id': float,      # Can contain 'nan' values.
        'origin-dataset': str,
        'origin-patient-id': str,
        'regions': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, TrainingDataset(d, check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    loaders = MultiLoader.build_loaders(datasets, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_subfolds=n_subfolds, shuffle_train=False, use_split_file=use_split_file, **kwargs)
    if n_folds is not None or use_split_file:
        if n_subfolds is not None:
            loader_names = ['train', 'validate', 'subtest', 'test']
        else:
            loader_names = ['train', 'validate', 'test']
    else:
        loader_names = ['train', 'validate']

    # Get values for this region.
    for loader, loader_name in zip(loaders, loader_names):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                sample = dataset_map[dataset].sample(sample_id)
                group_id = sample.group_id
                origin_ds, origin_pat_id = sample.origin
                regions = ','.join(sample.list_regions())
                data = {
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'group-id': group_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id,
                    'regions': regions
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def create_multi_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    load_all_samples: bool = False,
    n_folds: Optional[int] = None,
    region: Optional[PatientRegions] = None,
    test_fold: Optional[int] = None,
    use_split_file: bool = False,
    **kwargs) -> None:
    logging.arg_log('Creating multi-loader manifest', ('dataset', 'check_processed', 'n_folds', 'test_fold'), (dataset, check_processed, n_folds, test_fold))
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)

    # Get regions if 'None'.
    if regions is None:
        regions = []
        for dataset in datasets: 
            set_regions = TrainingDataset(dataset).list_regions()
            regions += set_regions
        regions = list(sorted(np.unique(regions)))

    # Get manifest.
    df = get_multi_loader_manifest(datasets, check_processed=check_processed, load_all_samples=load_all_samples, n_folds=n_folds, region=regions, test_fold=test_fold, **kwargs)

    # Save manifest.
    filepath = os.path.join(config.directories.reports, 'loader-manifests', encode(datasets), encode(regions), f'load-all-samples-{load_all_samples}-n-folds-{n_folds}-test-fold-{test_fold}-use-split-file-{use_split_file}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_multi_loader_manifest(
    dataset: Union[str, List[str]],
    load_all_samples: bool = False,
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    region: Optional[PatientRegions] = None,
    test_fold: Optional[int] = None,
    test_subfold: Optional[int] = None,
    use_split_file: bool = False) -> pd.DataFrame:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)

    # Get regions if 'None'.
    if regions is None:
        regions = []
        for dataset in datasets: 
            set_regions = TrainingDataset(dataset).list_regions()
            regions += set_regions
        regions = list(sorted(np.unique(regions)))

    # Load file.
    filepath = os.path.join(config.directories.reports, 'loader-manifests', encode(datasets), encode(regions), f'load-all-samples-{load_all_samples}-n-folds-{n_folds}-test-fold-{test_fold}-use-split-file-{use_split_file}.csv')
    df = pd.read_csv(filepath)
    df = df.astype({ 'origin-patient-id': str, 'sample-id': str })

    return df

def create_multi_loader_figures(
    dataset: Union[str, List[str]],
    n_folds: Optional[int] = None,
    n_subfolds: Optional[int] = None,
    random_seed: float = 42,
    region: Optional[PatientRegions] = None,
    test_fold: Optional[int] = None,
    test_subfold: Optional[int] = None,
    use_augmentation: bool = False,
    use_split_file: bool = False) -> None:
    logging.arg_log('Creating loader figures', ('dataset', 'region'), (dataset, region))
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)

    # Get regions if 'None'.
    if regions is None:
        regions = []
        for dataset in datasets: 
            set_regions = TrainingDataset(dataset).list_regions()
            regions += set_regions
        regions = list(sorted(np.unique(regions)))

    # Create transforms.
    if use_augmentation:
        seed_everything(random_seed, workers=True)      # Ensure reproducible augmentation.
        train_transform, val_transform = get_transforms()
    else:
        train_transform = None
        val_transform = None

    # Create loaders.
    train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, batch_size=1, n_folds=n_folds, region=regions, shuffle_train=False, test_fold=test_fold, transform_train=train_transform, transform_val=val_transform, use_split_file=use_split_file)

    # loaders = (train_loader, val_loader, test_loader)
    loaders = (train_loader, val_loader)

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    # Create PDF.
    pdf = FPDF()
    pdf.set_section_title_styles(
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=24,
            color=0,
            t_margin=3,
            l_margin=12,
            b_margin=0
        ),
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=18,
            color=0,
            t_margin=12,
            l_margin=12,
            b_margin=0
        ),
        TitleStyle(
            font_family='Times',
            font_style='B',
            font_size_pt=12,
            color=0,
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # names = ('train', 'val', 'test')
    names = ('train', 'val')
    for loader, name in zip(loaders, names):
        # Start sample section.
        pdf.add_page()
        pdf.start_section(f'Loader: {name}')

        logging.info(f"Creating '{name}' loader figures.")
        for i, (desc_b, x_b, y_b, mask_b, weights_b) in enumerate(tqdm(iter(loader))):
            ct_data = x_b[0, 0]
            size = ct_data.shape
            spacing = TrainingDataset(datasets[0]).params['output-spacing']
            region_data = dict((r, y_b[0, i + 1]) for i, r in enumerate(regions))

            # Start sample section.
            if i != 0:
                pdf.add_page()
            pdf.start_section(f'Sample: {desc_b[0]}', level=1)

            for i, region in enumerate(regions):
                if not mask_b[0, i + 1]:
                    continue

                label = y_b[0, i + 1].numpy()
                region_data = { region: label }
                
                # Start region section.
                if i != 0:
                    pdf.add_page()
                pdf.start_section(f'Region: {region}', level=2)

                views = [0, 1, 2]
                img_coords = (
                    (img_l_margin, img_t_margin),
                    (img_l_margin + img_width, img_t_margin),
                    (img_l_margin, img_t_margin + img_height)
                )
                for view, page_coord in zip(views, img_coords):
                    # Set figure.
                    filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                    plot_patient(desc_b[0], size, spacing, centre_of=region, ct_data=x_b[0, 0].numpy(), region_data=region_data, savepath=filepath, show=False, show_extent=True, view=view)

                    # Add image to report.
                    pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                    # Delete temp file.
                    os.remove(filepath)

    # Save PDF.
    filename = 'figures-aug.pdf' if use_augmentation else 'figures.pdf'
    filepath = os.path.join(config.directories.reports, 'loader-figures', encode(datasets), encode(regions), filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')
 