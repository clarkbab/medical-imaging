from fpdf import FPDF, TitleStyle
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import DataFrame
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage.measurements import label as label_objects
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid1

from mymi import config
from mymi.dataset import DicomDataset, NiftiDataset
from mymi.evaluation.dataset.nifti import load_localiser_evaluation, load_segmenter_evaluation
from mymi.geometry import get_extent, get_extent_centre, get_extent_width_mm
from mymi.loaders import Loader
from mymi import logging
from mymi.metrics import mean_intensity, snr
from mymi.models.systems import Localiser
from mymi.plotting.dataset.nifti import plot_localiser_prediction, plot_patient, plot_registration, plot_segmenter_prediction, plot_multi_segmenter_prediction, plot_adaptive_segmenter_prediction
from mymi.postprocessing import largest_cc_3D, get_object, one_hot_encode
from mymi.regions import regions_to_list as regions_to_list
from mymi.types import Axis, ModelName, PatientRegion, PatientRegions
from mymi.utils import append_row, arg_to_list, encode

def get_region_overlap_summary(
    dataset: str,
    region: str) -> DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(labels='all', region=region)

    cols = {
        'dataset': str,
        'patient-id': str,
        'region': str,
        'n-overlap': int
    }
    df = DataFrame(columns=cols.keys())

    for pat_id in tqdm(pat_ids):
        pat = set.patient(pat_id)
        if not pat.has_region(region, labels='all'):
            continue

        # Load region data.
        region_data = pat.region_data(labels='all')

        # Calculate overlap for other regions.
        for r in region_data.keys():
            if r == region:
                continue

            n_overlap = (region_data[region] & region_data[r]).sum()
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'region': r,
                'n-overlap': n_overlap
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def get_region_summary(
    dataset: str,
    region: str,
    labels: Literal['included', 'excluded', 'all'] = 'included') -> DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(labels=labels, regions=region)

    cols = {
        'dataset': str,
        'patient-id': str,
        'metric': str,
        'value': float
    }
    df = DataFrame(columns=cols.keys())

    for pat_id in tqdm(pat_ids):
        pat = set.patient(pat_id)
        ct_data = pat.ct_data
        spacing = pat.ct_spacing
        label = pat.region_data(labels=labels, regions=region)[region]

        data = {
            'dataset': dataset,
            'patient-id': pat_id,
        }

        # Add 'min/max' extent metrics.
        min_extent_vox = np.argwhere(label).min(axis=0)
        min_extent_mm = min_extent_vox * spacing
        max_extent_vox = np.argwhere(label).max(axis=0)
        max_extent_mm = max_extent_vox * spacing
        for axis, min, max in zip(('x', 'y', 'z'), min_extent_mm, max_extent_mm):
            data['metric'] = f'min-extent-mm-{axis}'
            data['value'] = min
            df = append_row(df, data)
            data['metric'] = f'max-extent-mm-{axis}'
            data['value'] = max
            df = append_row(df, data)

        # Add 'connected' metrics.
        data['metric'] = 'connected'
        lcc_label = largest_cc_3D(label)
        data['value'] = 1 if lcc_label.sum() == label.sum() else 0
        df = append_row(df, data)
        data['metric'] = 'connected-largest-p'
        data['value'] = lcc_label.sum() / label.sum()
        df = append_row(df, data)

        # Add intensity metrics.
        if pat.has_region('Brain'):
            data['metric'] = 'snr-brain'
            brain_label = pat.region_data(regions='Brain')['Brain']
            data['value'] = snr(ct_data, label, brain_label, spacing)
            df = append_row(df, data)
        data['metric'] = 'mean-intensity'
        data['value'] = mean_intensity(ct_data, label)
        df = append_row(df, data)

        # Add OAR extent.
        ext_width_mm = get_extent_width_mm(label, spacing)
        if ext_width_mm is None:
            ext_width_mm = (0, 0, 0)
        data['metric'] = 'extent-mm-x'
        data['value'] = ext_width_mm[0]
        df = append_row(df, data)
        data['metric'] = 'extent-mm-y'
        data['value'] = ext_width_mm[1]
        df = append_row(df, data)
        data['metric'] = 'extent-mm-z'
        data['value'] = ext_width_mm[2]
        df = append_row(df, data)

        # Add extent of largest connected component.
        extent = get_extent(lcc_label)
        if extent:
            min, max = extent
            extent_vox = np.array(max) - min
            extent_mm = extent_vox * spacing
        else:
            extent_mm = (0, 0, 0)
        data['metric'] = 'connected-extent-mm-x'
        data['value'] = extent_mm[0]
        df = append_row(df, data)
        data['metric'] = 'connected-extent-mm-y'
        data['value'] = extent_mm[1]
        df = append_row(df, data)
        data['metric'] = 'connected-extent-mm-z'
        data['value'] = extent_mm[2]
        df = append_row(df, data)

        # Add volume.
        vox_volume = reduce(np.multiply, spacing)
        data['metric'] = 'volume-mm3'
        data['value'] = vox_volume * label.sum() 
        df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_counts(dataset: str) -> None:
    pr_df = get_region_counts(dataset)
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pr_df.to_csv(filepath, index=False)

def create_region_contrast_report(
    dataset: str,
    region: PatientRegion,
    noise_region: PatientRegion = 'Brain') -> None:
    logging.arg_log('Creating region contrast report', ('dataset', 'region', 'noise_region'), (dataset, region, noise_region))

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str,
        'hu-oar': float,
        'hu-margin': float,
        'contrast': float,
        'noise': float,
        f'noise-{noise_region.lower()}': float,
        'cnr': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Load data.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(region=region)

    for pat_id in tqdm(pat_ids):
        # Load CT and region data.
        pat = set.patient(pat_id)
        if not pat.has_region(region):
            continue
        ct_data = pat.ct_data
        region_data = pat.region_data(region=region)[region]

        # Get OAR label and margin label.
        region_data_margin = np.logical_xor(binary_dilation(region_data, iterations=3), region_data)

        # Get OAR and margin HU values.
        hu_oar = ct_data[np.nonzero(region_data)]
        hu_oar_mean = hu_oar.mean()
        hu_margin = ct_data[np.nonzero(region_data_margin)]
        hu_margin_mean = hu_margin.mean()

        # Calculate contrast-to-noise (CNR) ratio.
        contrast = hu_oar_mean - hu_margin_mean
        background_noise = hu_oar.std()
        cnr = contrast / background_noise

        # Calculate region noise.
        if pat.has_region(noise_region):
            noise_data = pat.region_data(region=noise_region)[noise_region]
            noise_data_eroded = binary_erosion(noise_data, iterations=3)
            if noise_data_eroded.sum() == 0:
                raise ValueError(f"Eroded noise data for region '{noise_region}' is empty, choose a larger region.")
            hu_noise = ct_data[np.nonzero(noise_data_eroded)]
            region_noise = hu_noise.std()
        else:
            region_noise = np.nan

        # Add data.
        data = {
            'patient-id': pat_id,
            'region': region,
            'hu-oar': hu_oar_mean,
            'hu-margin': hu_margin_mean,
            'contrast': contrast,
            'noise': background_noise,
            f'noise-{noise_region.lower()}': region_noise,
            'cnr': cnr
        }
        df = append_row(df, data)
            
    # Save report.
    df = df.astype(cols)
    filepath = os.path.join(set.path, 'reports', 'region-contrast', f'{region}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_region_overlap_summary(
    dataset: str,
    region: str) -> None:
    logging.arg_log('Creating region overlap summary', ('dataset', 'region'), (dataset, region))

    # Generate counts report.
    df = get_region_overlap_summary(dataset, region)

    # Save report.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-overlap-summaries', f'{region}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_region_summary(
    dataset: str,
    regions: Optional[PatientRegions] = None) -> None:
    logging.arg_log('Creating region summaries', ('dataset', 'regions'), (dataset, regions))
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions)
    if regions is None:
        regions = set.list_regions()

    for region in tqdm(regions):
        # Check if there are patients with this region.
        n_pats = len(set.list_patients(labels='all', regions=region))
        if n_pats == 0:
            # Allows us to pass all regions from Spartan 'array' job.
            logging.error(f"No patients with region '{region}' found for dataset '{set}'.")
            return

        # Generate counts report.
        df = get_region_summary(dataset, region, labels='all')

        # Save report.
        filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def get_region_counts(dataset: str) -> DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str
    }
    df = DataFrame(columns=cols.keys())

    # Add rows.
    for pat_id in tqdm(pat_ids):
        pat_regions = set.patient(pat_id).list_regions()
        for pat_region in pat_regions:
            data = {
                'patient-id': pat_id,
                'region': pat_region
            }
            df = append_row(df, data)

    return df

def load_region_counts(
    dataset: str,
    regions: PatientRegions = 'all',
    exists_only: bool = False) -> Union[DataFrame, bool]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
        else:
            df = pd.read_csv(filepath)
            if regions != 'all':
                regions = regions_to_list(regions)
                df = df[df['region'].isin(regions)]
            return df
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Patient regions report doesn't exist for dataset '{dataset}'.")

def load_plan_labels_report(dataset: str) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'plan-labels.csv')
    df = pd.read_csv(filepath)
    return df

def _get_outlier_cols_func(
    lim_df: DataFrame) -> Callable[[pd.Series], Dict]:
    # Create function to operate on row of 'region summary' table.
    def _outlier_cols(row: pd.Series) -> Dict:
        data = {}

        # Add outlier info.
        for column in lim_df.index:
            col_stats = lim_df.loc[column]
            if row[column] < col_stats['low']:
                outlier = True
                outlier_dir = 'low'
                if col_stats['iqr'] == 0:
                    outlier_n_iqr = np.inf
                else:
                    outlier_n_iqr = (col_stats['q1'] - row[column]) / col_stats['iqr']
            elif row[column] > col_stats['high']:
                outlier = True
                outlier_dir = 'high'
                if col_stats['iqr'] == 0:
                    outlier_n_iqr = np.inf
                else:
                    outlier_n_iqr = (row[column] - col_stats['q3']) / col_stats['iqr']
            else:
                outlier = False
                outlier_dir = ''
                outlier_n_iqr = np.nan

            data[f'{column}-out'] = outlier
            data[f'{column}-out-dir'] = outlier_dir
            data[f'{column}-out-num-iqr'] = outlier_n_iqr

        return data
    return _outlier_cols

def add_region_summary_outliers(
    df: DataFrame,
    columns: List[str]) -> DataFrame:

    # Get outlier limits.
    q1 = df.quantile(0.25, numeric_only=True)[columns].rename('q1')
    q3 = df.quantile(0.75, numeric_only=True)[columns].rename('q3')
    iqr = (q3 - q1).rename('iqr')
    low = (q1 - 1.5 * iqr).rename('low')
    high = (q3 + 1.5 * iqr).rename('high')
    lim_df = pd.concat([q1, q3, iqr, low, high], axis=1)

    # Add columns.
    func = _get_outlier_cols_func(lim_df)
    out_df = df.apply(func, axis=1, result_type='expand')
    df = pd.concat([df, out_df], axis=1)
    return df

def load_region_overlap_summary(
    dataset: str,
    region: str,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    raise_error: bool = True) -> Optional[pd.DataFrame]:

    # Load summary.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-overlap-summaries', f'{region}.csv')
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Summary not found for region '{region}', dataset '{set}'.")
        else:
            return None
    df = pd.read_csv(filepath)

    # Filter by 'excluded-labels.csv'.
    exc_df = set.excluded_labels
    if labels != 'all':
        if exc_df is None:
            raise ValueError(f"No 'excluded-labels.csv' specified for '{set}', should pass labels='all'.")
    if labels == 'included':
        df = df.merge(exc_df, on=['patient-id', 'region'], how='left', indicator=True)
        df = df[df._merge == 'left_only'].drop(columns='_merge')
    elif labels == 'excluded':
        df = df.merge(exc_df, on=['patient-id', 'region'], how='left', indicator=True)
        df = df[df._merge == 'both'].drop(columns='_merge')

    return df

def load_region_summary(
    dataset: Union[str, List[str]],
    labels: Literal['included', 'excluded', 'all'] = 'included',
    regions: Optional[PatientRegions] = None,
    pivot: bool = False,
    raise_error: bool = True) -> Optional[pd.DataFrame]:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(regions)

    # Load summary.
    dfs = []
    for dataset in datasets:
        # Get regions if 'None'.
        set = NiftiDataset(dataset)
        exc_df = set.excluded_labels
        if regions is None:
            dataset_regions = set.list_regions()
        else:
            dataset_regions = regions

        for region in dataset_regions:
            filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
            if not os.path.exists(filepath):
                if raise_error:
                    raise ValueError(f"Summary not found for region '{region}', dataset '{set}'.")
                else:
                    # Skip this region.
                    continue

            # Add CSV.
            df = pd.read_csv(filepath, dtype={ 'patient-id': str })
            df.insert(1, 'region', region)

            # Filter by 'excluded-labels.csv'.
            rexc_df = None if exc_df is None else exc_df[exc_df['region'] == region] 
            if labels != 'all':
                if rexc_df is None:
                    raise ValueError(f"No 'excluded-labels.csv' specified for '{set}', should pass labels='all'.")
            if labels == 'included':
                df = df.merge(rexc_df, on=['patient-id', 'region'], how='left', indicator=True)
                df = df[df._merge == 'left_only'].drop(columns='_merge')
            elif labels == 'excluded':
                df = df.merge(rexc_df, on=['patient-id', 'region'], how='left', indicator=True)
                df = df[df._merge == 'both'].drop(columns='_merge')

            # Add region summary.
            dfs.append(df)

    # Concatenate loaded files.
    if len(dfs) == 0:
        return None
    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)

    # If pivot.
    if pivot:
        df = df.pivot(index=['dataset', 'patient-id', 'region'], columns='metric', values='value')
        df = df.reset_index()

    return df

def load_region_contrast_report(
    dataset: Union[str, List[str]],
    region: PatientRegions) -> pd.DataFrame:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
            
    # Load reports.
    dfs = []
    for dataset in datasets:
        set = NiftiDataset(dataset)

        for region in regions:
            filepath = os.path.join(set.path, 'reports', 'region-contrast', f'{region}.csv')
            df = pd.read_csv(filepath)
            df.insert(0, 'dataset', dataset)
            dfs.append(df)

    # Concatenate reports.
    df = pd.concat(dfs, axis=0)

    return df

def get_ct_info_summary(
    dataset: str,
    region: Optional[PatientRegions] = None) -> pd.DataFrame:
    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(region=region)

    cols = {
        'dataset': str,
        'patient-id': str,
        'metric': str,          # e.g. 'manufacturer', 'study-date'
        'value': str
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat_id in tqdm(pat_ids):
        pat = set.patient(pat_id)
        origin_dataset, origin_pat_id, origin_study_id = pat.origin
        study = DicomDataset(origin_dataset).patient(origin_pat_id).study(origin_study_id)

        # Add contrast details.
        ct = study.first_ct
        data = {
            'dataset': dataset,
            'patient-id': pat_id,
            'metric': 'contrast',
            'value': getattr(ct, 'ContrastBolusAgent', '')
        }
        df = append_row(df, data)

        # Add 'kVp', 'mA'.
        metrics = ['kvp', 'ma', 'exposure-ms']
        attrs = ['KVP', 'XRayTubeCurrent', 'ExposureTime']
        for metric, attr in zip(metrics, attrs):
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'metric': metric,
                'value': getattr(ct, attr, '')
            }
            df = append_row(df, data)

        # Add manufacturer details.
        metrics = ['manufacturer-make', 'manufacturer-model']
        attrs = ['Manufacturer', 'ManufacturerModelName']
        for metric, attr in zip(metrics, attrs):
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'metric': metric,
                'value': getattr(ct, attr, '')
            }
            df = append_row(df, data)

        # Add study date.
        data = {
            'dataset': dataset,
            'patient-id': pat_id,
            'metric': 'study-date',
            'value': study.date
        }
        df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def get_ct_summary(dataset: str) -> pd.DataFrame:
    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    cols = {
        'dataset': str,
        'patient-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat_id in tqdm(pat_ids):
        # Load values.
        patient = set.patient(pat_id)
        size = patient.ct_size
        spacing = patient.ct_spacing

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df
    
def create_ct_info_summary_report(
    dataset: str,
    region: Optional[PatientRegions] = None) -> None:
    # Get regions.
    set = NiftiDataset(dataset)
    regions = arg_to_list(region, str) if region is None else set.list_regions()

    # Get summary.
    df = get_ct_info_summary(dataset, region=regions)

    # Save summary.
    filepath = os.path.join(set.path, 'reports', 'ct-info-summary', encode(regions), 'ct-summary.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_ct_summary(dataset: str) -> None:
    # Get regions.
    set = NiftiDataset(dataset)

    # Get summary.
    df = get_ct_summary(dataset)

    # Save summary.
    filepath = os.path.join(set.path, 'reports', 'ct-summary.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_info_summary_report(
    dataset: str,
    region: Optional[PatientRegions] = None) -> pd.DataFrame:
    # Get regions.
    set = NiftiDataset(dataset)
    regions = arg_to_list(region, str) if region is None else set.list_regions()

    filepath = os.path.join(set.path, 'reports', 'ct-info-summary', encode(regions), 'ct-summary.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"CT info summary report doesn't exist for dataset '{dataset}'.")
    return pd.read_csv(filepath)

def load_ct_summary(datasets: Union[str, List[str]]) -> pd.DataFrame:
    datasets = arg_to_list(datasets, str)

    dfs = []
    for dataset in datasets:
        # Get regions.
        set = NiftiDataset(dataset)

        filepath = os.path.join(set.path, 'reports', 'ct-summary.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"CT summary report doesn't exist for dataset '{dataset}'.")
        df = pd.read_csv(filepath)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.astype({ 'patient-id': str })

    return df

def create_multi_segmenter_heatmap_figures(
    dataset: Union[str, List[str]],
    model: ModelName,
    region: str,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    view: Union[Axis, List[Axis]] = list(range(3))) -> None:
    datasets = arg_to_list(dataset, str)
    views = arg_to_list(view, str)
    logging.info(f"Creating segmenter prediction figures for datasets '{datasets}', region '{region}', test fold '{test_fold}', localiser '{localiser}', segmenter '{segmenter}' and view '{view}'.")

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 150
    img_height = 200

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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Add patient.
            pdf.add_page()
            pdf.start_section(f'{dataset} - {pat_id}')

            # Create images.
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Add image to report.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plot_segmenter_prediction(dataset, pat_id, localiser, segmenter, centre=region, crop=region, savepath=filepath, show=False, show_legend=False, view=view)
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)
                os.remove(filepath)

    # Save PDF.
    # We have to 'encode' localisers/segmenters because they could be a list of models.
    filepath = os.path.join(config.directories.reports, 'prediction-figures', 'segmenter', encode(datasets), region, encode(localiser), encode(segmenter), f'figures-fold-{test_fold}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_plan_labels_report(dataset: str) -> None:
    logging.arg_log('Creating plan labels report', ('dataset',), (dataset,))

    nifti_set = NiftiDataset(dataset)
    index = nifti_set.index
    dicom_set = DicomDataset(dataset)
    def get_rtplan_label(row: pd.Series):
        study = dicom_set.patient(row['origin-patient-id']).study(row['origin-study-id'])
        rtplan = study.default_rtplan
        if rtplan is None:
            return ''
        plan_label = rtplan.get_rtplan().RTPlanLabel
        return plan_label
    tqdm.pandas()   # Enable 'progress_apply' method.
    index['rtplan-label'] = index.progress_apply(get_rtplan_label, axis=1)

    filepath = os.path.join(nifti_set.path, 'reports', 'plan-labels.csv')
    index.to_csv(filepath, index=False)

def create_totalseg_prediction_figures(dataset: str) -> None:
    # We just saved totalseg predictions as regions in the NiFTi dataset because the names
    # didn't overlap with nay existing regions.
    logging.info(f"Creating totalseg prediction figures for dataset '{dataset}'.")

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 180
    img_height = 200

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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Load patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    centre_regions = [
        'SpinalCord',
        'Spinal_Cord'
    ]

    # Add images.
    for pat_id in tqdm(pat_ids):
        # Get centre region.
        pat = set.patient(pat_id)
        centre = None
        for r in centre_regions:
            if pat.has_region(r):
                centre = r
                break
        if centre is None:
            raise ValueError(f"Patient '{pat_id}' doesn't have one of the required regions: {centre_regions}. Please choose from: {pat.list_regions()}.")

        # Add patient.
        pdf.add_page()
        pdf.start_section(f'{dataset} - {pat_id}')

        # Create images.
        # views = list(range(3))
        views = [1]
        img_coords = (
            (img_l_margin, img_t_margin),
            # (img_l_margin + img_width, img_t_margin),
            # (img_l_margin, img_t_margin + img_height)
        )
        for view, page_coord in zip(views, img_coords):
            # Add image to report.
            filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            kwargs = dict(
                centre=centre,
                region='all',
                savepath=filepath,
                show=False,
                show_legend=True,
                view=view,
                window='bone'
            )
            plot_patient(dataset, pat_id, **kwargs)
            pdf.image(filepath, *page_coord, w=img_width, h=img_height)
            os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'region-figures', 'totalseg.pdf')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_multi_segmenter_prediction_figures(
    dataset: str,
    region: str,
    model: Union[ModelName, List[ModelName]],
    model_region: PatientRegions,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    view: Union[Axis, List[Axis]] = list(range(3)),
    **kwargs) -> None:
    views = arg_to_list(view, str)
    logging.info(f"Creating multi-segmenter prediction figures for dataset '{dataset}', region '{region}', test fold '{test_fold}', model '{model}' and view '{view}'.")

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(dataset, region, n_folds=n_folds, test_fold=test_fold, **kwargs)

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 150
    img_height = 200

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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Add patient.
            pdf.add_page()
            pdf.start_section(f'{dataset} - {pat_id}')

            # Create images.
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Add image to report.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plot_multi_segmenter_prediction(dataset, pat_id, model, model_region, centre=region, crop=region, savepath=filepath, show=False, show_legend=False, view=view, **kwargs)
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)
                os.remove(filepath)

    # Save PDF.
    # We have to 'encode' localisers/segmenters because they could be a list of models.
    set = NiftiDataset(dataset)
    filepath = os.path.join(NiftiDataset(set).path, 'reports', 'prediction-figures', 'multi-segmenter', *model, f'figures-fold-{test_fold}.pdf')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_segmenter_prediction_figures(
    dataset: Union[str, List[str]],
    region: str,
    localiser: Union[ModelName, List[ModelName]],
    segmenter: Union[ModelName, List[ModelName]],
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None,
    view: Union[Axis, List[Axis]] = list(range(3))) -> None:
    datasets = arg_to_list(dataset, str)
    views = arg_to_list(view, str)
    logging.info(f"Creating segmenter prediction figures for datasets '{datasets}', region '{region}', test fold '{test_fold}', localiser '{localiser}', segmenter '{segmenter}' and view '{view}'.")

    # Create test loader.
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Set PDF margins.
    img_t_margin = 35
    img_l_margin = 5
    img_width = 150
    img_height = 200

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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Make predictions.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')

            # Add patient.
            pdf.add_page()
            pdf.start_section(f'{dataset} - {pat_id}')

            # Create images.
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Add image to report.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plot_segmenter_prediction(dataset, pat_id, localiser, segmenter, centre=region, crop=region, savepath=filepath, show=False, show_legend=False, view=view)
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)
                os.remove(filepath)

    # Save PDF.
    # We have to 'encode' localisers/segmenters because they could be a list of models.
    filepath = os.path.join(config.directories.reports, 'prediction-figures', 'segmenter', encode(datasets), region, encode(localiser), encode(segmenter), f'figures-fold-{test_fold}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_region_figures(
    dataset: str,
    region: str,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    subregions: bool = False,
    **kwargs) -> None:
    logging.arg_log('Creating region figures', ('dataset', 'region', 'labels', 'subregions'), (dataset, region, labels, subregions))

    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(labels=labels, region=region)

    # Get excluded regions.
    exc_df = set.excluded_labels
    if exc_df is None and labels != 'all':
        raise ValueError("'excluded-labels.csv' must be present to split included from excluded regions.")

    # Keep regions with patients.
    df = load_region_summary(dataset, labels=labels, region=region)
    df = df.pivot(index=['dataset', 'region', 'patient-id'], columns='metric', values='value').reset_index()

    # Add 'extent-mm' outlier info.
    columns = ['extent-mm-x', 'extent-mm-y', 'extent-mm-z']
    df = add_region_summary_outliers(df, columns)

    # Set PDF margins.
    img_t_margin = 35
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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    for pat_id in tqdm(pat_ids):
        # Skip on inclusion/exclusion criteria.
        if exc_df is not None:
            edf = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
            if labels == 'included' and len(edf) >= 1:
                logging.info(f"Patient '{pat_id}' skipped. Excluded by 'excluded-labels.csv' and only 'included' labels shown in report.")
                continue
            elif labels == 'excluded' and len(edf) == 0:
                logging.info(f"Patient '{pat_id}' skipped. Not excluded by 'excluded-labels.csv' and only 'excluded' labels shown in report.")
                continue

        # Add patient.
        pdf.add_page()
        pdf.start_section(pat_id)

        # Add region info.
        pdf.start_section('Region Info', level=1)

        # Create table.
        table_t_margin = 50
        table_l_margin = 12
        table_line_height = 2 * pdf.font_size
        table_col_widths = (15, 35, 30, 45, 45)
        print(df.head())
        print(pat_id)
        pat_info = df[df['patient-id'] == pat_id].iloc[0]
        table_data = [('Axis', 'Extent [mm]', 'Outlier', 'Outlier Direction', 'Outlier Num. IQR')]
        for axis in ['x', 'y', 'z']:
            colnames = {
                'extent': f'extent-mm-{axis}',
                'extent-out': f'extent-mm-{axis}-out',
                'extent-out-dir': f'extent-mm-{axis}-out-dir',
                'extent-out-num-iqr': f'extent-mm-{axis}-out-num-iqr'
            }
            n_iqr = pat_info[colnames['extent-out-num-iqr']]
            format = '.2f' if n_iqr and n_iqr != np.nan else ''
            table_data.append((
                axis,
                f"{pat_info[colnames['extent']]:.2f}",
                str(pat_info[colnames['extent-out']]),
                pat_info[colnames['extent-out-dir']],
                f"{pat_info[colnames['extent-out-num-iqr']]:{format}}",
            ))

        for i, row in enumerate(table_data):
            if i == 0:
                pdf.set_font('Helvetica', 'B', 12)
            else:
                pdf.set_font('Helvetica', '', 12)
            pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
            for j, value in enumerate(row):
                pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add subregion info.
        if subregions:
            # Get object info.
            obj_df = get_object_summary(dataset, pat_id, region)

            if len(obj_df) > 1:
                pdf.start_section('Subregion Info', level=1)

                # Create table.
                table_t_margin = 105
                table_l_margin = 12
                table_line_height = 2 * pdf.font_size
                table_col_widths = (15, 35, 30, 45, 45)
                table_data = [('ID', 'Volume [mm^3]', 'Volume [prop.]', 'Extent Centre [vox]', 'Extent Width [mm]')]
                for i, row in obj_df.iterrows():
                    table_data.append((
                        str(i),
                        f"{row['volume-mm3']:.2f}",
                        f"{row['volume-p-total']:.2f}",
                        row['extent-centre-vox'],
                        str(tuple([round(e, 2) for e in eval(row['extent-mm'])]))
                    ))
                for i, row in enumerate(table_data):
                    if i == 0:
                        pdf.set_font('Helvetica', 'B', 12)
                    else:
                        pdf.set_font('Helvetica', '', 12)
                    pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
                    for j, value in enumerate(row):
                        pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add region images.
        pdf.add_page()
        pdf.start_section(f'Region Images', level=1)

        # Create images.
        views = list(range(3))
        img_coords = (
            (img_l_margin, img_t_margin),
            (img_l_margin + img_width, img_t_margin),
            (img_l_margin, img_t_margin + img_height)
        )
        for view, page_coord in zip(views, img_coords):
            # Set figure.
            savepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plot_patient(dataset, pat_id, centre=region, colour=['y'], crop=region, labels=labels, region=region, show_extent=True, savepath=savepath, view=view, **kwargs)

            # Add image to report.
            pdf.image(savepath, *page_coord, w=img_width, h=img_height)
            os.remove(savepath)

        # Add subregion images.
        if subregions and len(obj_df) > 1:
            for i, row in obj_df.iterrows():
                pdf.add_page()
                pdf.start_section(f'Subregion {i} Images', level=1)

                # Create images.
                views = list(range(3))
                img_coords = (
                    (img_l_margin, img_t_margin),
                    (img_l_margin + img_width, img_t_margin),
                    (img_l_margin, img_t_margin + img_height)
                )
                for view, page_coord in zip(views, img_coords):
                    # Set figure.
                    def postproc(a: np.ndarray):
                        return get_object(a, i)
                    plot_patient(dataset, pat_id, centre=region, colours=['y'], postproc=postproc, labels=labels, region=region, show_extent=True, view=view, **kwargs)

                    # Save temp file.
                    filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                    plt.savefig(filepath)
                    plt.close()

                    # Add image to report.
                    pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                    # Delete temp file.
                    os.remove(filepath)

    # Save PDF.
    postfix = '' if labels == 'all' else f'-{labels}'
    filepath = os.path.join(set.path, 'reports', 'region-figures', f'{region}{postfix}.pdf')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_registration_figures_report(
    dataset: str,
    **kwargs) -> None:
    logging.arg_log('Creating registration report', ('dataset',), (dataset,))

    # Set PDF margins.
    img_t_margin = 2
    img_l_margin = 52
    img_width = 150
    img_height = 95

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
            t_margin=16,
            l_margin=12,
            b_margin=0
        )
    ) 

    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()
    n_pats = len(pat_ids) // 2

    for pat_id in tqdm(range(n_pats)):
        fixed_pat_id = f'{pat_id}-1'
        moving_pat_id = f'{pat_id}-0'

        # Add registration plots.
        pdf.add_page()
        pdf.start_section(f"Patient: {pat_id}")

        # Create axial image.
        views = list(range(3))
        img_coords = [
            (img_l_margin, img_t_margin),
            (img_l_margin, 2 * img_t_margin + img_height),
            (img_l_margin, 3 * img_t_margin + 2 * img_height),
        ]

        for view, (x, y) in zip(views, img_coords):
            # Save temporary figure.
            savepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plot_registration(dataset, fixed_pat_id, moving_pat_id, savepath=savepath, show=False, view=view, **kwargs)

            # Add image to report.
            pdf.image(savepath, x=x, y=y, w=img_width, h=img_height)
            os.remove(savepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'registration-figures', 'report.pdf')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def get_object_summary(
    dataset: str,
    pat_id: str,
    region: str) -> pd.DataFrame:
    # Get objects.
    pat = NiftiDataset(dataset).patient(pat_id)

    spacing = pat.ct_spacing
    label = pat.region_data(region=region)[region]
    objs, n_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = one_hot_encode(objs)
    
    cols = {
        'extent-centre-vox': str,
        'extent-mm': str,
        'extent-vox': str,
        'volume-mm3': float,
        'volume-p-total': float,
        'volume-vox': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    tot_voxels = label.sum()
    for i in range(n_objs):
        obj = objs[:, :, :, i]
        data = {}

        # Get extent.
        min, max = get_extent(obj)
        width = tuple(np.array(max) - min)
        width_mm = tuple(np.array(spacing) * width)
        data['extent-mm'] = str(width_mm)
        data['extent-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = get_extent_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        n_voxels = obj.sum()
        volume = n_voxels * vox_volume
        data['volume-vox'] = n_voxels
        data['volume-p-total'] = n_voxels / tot_voxels
        data['volume-mm3'] = volume
        df = append_row(df, data)

    df = df.astype(cols)
    return df

def create_localiser_figures(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str]) -> None:
    localiser = Localiser.replace_ckpt_aliases(*localiser)
    logging.info(f"Creating localiser figures for dataset '{dataset}', region '{region}' and localiser '{localiser}'.")

    # Get patients.
    set = NiftiDataset(dataset)
    pats = set.list_patients(region=region)

    # Exit if region not present.
    set_regions = list(sorted(set.list_regions().region.unique()))
    if region not in set_regions:
        logging.info(f"No region '{region}' present in dataset '{dataset}'.")

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
        )
    ) 
    
    # Get errors for the region based upon 'extent-dist-x/y/z' metrics.
    eval_df = load_localiser_evaluation(dataset, localiser)
    error_df = eval_df[eval_df.metric.str.contains('extent-dist-')]
    error_df = error_df[(error_df.value.isnull()) | (error_df.value > 0)]

    # Add errors section.
    pdf.add_page()
    pdf.start_section('Errors')

    # Add table.
    table_t_margin = 45
    table_l_margin = 12
    table_line_height = 2 * pdf.font_size
    table_col_widths = (40, 40, 40)
    table_data = [('Patient', 'Metric', 'Value')]
    for _, row in error_df.iterrows():
        table_data.append((
            row['patient-id'],
            row.metric,
            f'{row.value:.3f}'
        ))
    for i, row in enumerate(table_data):
        if i == 0:
            pdf.set_font('Helvetica', 'B', 12)
        else:
            pdf.set_font('Helvetica', '', 12)
        pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
        for j, value in enumerate(row):
            pdf.cell(table_col_widths[j], table_line_height, value, border=1)

    for pat in tqdm(pats, leave=False):
        # Skip if patient doesn't have region.
        patient = set.patient(pat)
        if not patient.has_region(region):
            continue

        # Start info section.
        pdf.add_page()
        pdf.start_section(pat)
        pdf.start_section('Info', level=1)

        # Add table.
        table_t_margin = 45
        table_l_margin = 12
        table_line_height = 2 * pdf.font_size
        table_col_widths = (40, 40)
        table_data = [('Metric', 'Value')]
        pat_eval_df = eval_df[eval_df['patient-id'] == pat]
        for _, row in pat_eval_df.iterrows():
            table_data.append((
                row.metric,
                f'{row.value:.3f}'
            ))
        for i, row in enumerate(table_data):
            if i == 0:
                pdf.set_font('Helvetica', 'B', 12)
            else:
                pdf.set_font('Helvetica', '', 12)
            pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
            for j, value in enumerate(row):
                pdf.cell(table_col_widths[j], table_line_height, value, border=1)

        # Add images.
        pdf.add_page()
        pdf.start_section('Images', level=1)

        # Save images.
        views = list(range(3))
        img_coords = (
            (img_l_margin, img_t_margin),
            (img_l_margin + img_width, img_t_margin),
            (img_l_margin, img_t_margin + img_height)
        )
        for view, page_coord in zip(views, img_coords):
            # Set figure.
            plot_localiser_prediction(dataset, pat, region, localiser, centre=region, show_extent=True, show_patch=True, view=view, window=(3000, 500))

            # Save temp file.
            filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plt.savefig(filepath)
            plt.close()

            # Add image to report.
            pdf.image(filepath, *page_coord, w=img_width, h=img_height)

            # Delete temp file.
            os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'localiser-figures', *localiser, f'{region}.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')
