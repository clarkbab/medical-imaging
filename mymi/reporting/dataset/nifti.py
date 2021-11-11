from fpdf import FPDF, TitleStyle
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from typing import Callable, Dict, List
from uuid import uuid1

from mymi import config
from mymi import dataset as ds
from mymi import logging
from mymi.plotter.dataset.nifti import plot_patient_regions
from mymi.postprocessing import get_extent, get_largest_cc
from mymi import types

def get_region_summary(
    dataset: str,
    regions: List[str]) -> pd.DataFrame:
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    cols = {
        'patient': str,
        'region': str,
        'connected': bool,
        'connected-p': float,
        'conneted-extent-mm-x': float,
        'conneted-extent-mm-y': float,
        'conneted-extent-mm-z': float,
        'extent-mm-x': float,
        'extent-mm-y': float,
        'extent-mm-z': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get spacing.
        spacing = set.patient(pat).ct_spacing()

        # Get region data.
        pat_regions = set.patient(pat).list_regions(whitelist=regions)
        rs_data = set.patient(pat).region_data(regions=pat_regions)

        # Add extents for all regions.
        for region in rs_data.keys():
            data = {
                'patient': pat,
                'region': region
            }

            # See if OAR is single structure.
            label = rs_data[region]
            lcc_label = get_largest_cc(label)
            data['connected'] = True if lcc_label.sum() == label.sum() else False
            data['connected-p'] = lcc_label.sum() / label.sum()

            # Add OAR extent.
            extent = get_extent(label)
            if extent:
                min, max = extent
                extent_vox = np.array(max) - min
                extent_mm = extent_vox * spacing
            else:
                extent_mm = (0, 0, 0)
            data['extent-mm-x'] = extent_mm[0]
            data['extent-mm-y'] = extent_mm[1]
            data['extent-mm-z'] = extent_mm[2]

            # Add extent of largest connected component.
            extent = get_extent(lcc_label)
            if extent:
                min, max = extent
                extent_vox = np.array(max) - min
                extent_mm = extent_vox * spacing
            else:
                extent_mm = (0, 0, 0)
            data['connected-extent-mm-x'] = extent_mm[0]
            data['connected-extent-mm-y'] = extent_mm[1]
            data['connected-extent-mm-z'] = extent_mm[2]

            df = df.append(data, ignore_index=True)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_summary(
    dataset: str,
    regions: List[str]) -> None:
    logging.info(f"Creating region summary for dataset '{dataset}', regions '{regions}'...")

    # Generate counts report.
    df = get_region_summary(dataset, regions)

    # Save report.
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'region-summary-{hash}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def _get_outlier_cols_func(
    columns: List[str],
    lim_df: pd.DataFrame) -> Callable[[pd.Series], Dict]:
    def _outlier_cols(row: pd.Series) -> Dict:
        reg_stats = lim_df.loc[row.region]
        data = {}

        # Add outlier info.
        for column in columns:
            if row[column] < reg_stats[f'{column}-low']:
                outlier = True
                outlier_dir = 'LOW'
                if reg_stats[f'{column}-iqr'] != 0:
                    outlier_iqr = (reg_stats[f'{column}-q1'] - row[column]) / reg_stats[f'{column}-iqr']
                else:
                    outlier_iqr = np.inf
            elif row[column] > reg_stats[f'{column}-high']:
                outlier = True
                outlier_dir = 'HIGH'
                if reg_stats[f'{column}-iqr'] != 0:
                    outlier_iqr = (row[column] - reg_stats[f'{column}-q3']) / reg_stats[f'{column}-iqr']
                else:
                    outlier_iqr = np.inf
            else:
                outlier = False
                outlier_dir = ''
                outlier_iqr = np.nan

            data[f'{column}-out'] = outlier
            data[f'{column}-out-dir'] = outlier_dir
            data[f'{column}-out-iqr'] = outlier_iqr

        return data
    return _outlier_cols

def add_region_summary_outliers(
    df: pd.DataFrame,
    columns: List[str]) -> pd.DataFrame:

    # Get outlier limits.
    q1 = df.groupby(['region']).quantile(0.25)[columns]
    q3 = df.groupby(['region']).quantile(0.75)[columns]
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    # Rename columns.
    def colmap(cols, suffix):
        return dict(((c, f'{c}-{suffix}') for c in cols))
    q1 = q1.rename(columns=colmap(columns, 'q1'))
    q3 = q3.rename(columns=colmap(columns, 'q3'))
    iqr = iqr.rename(columns=colmap(columns, 'iqr'))
    low = low.rename(columns=colmap(columns, 'low'))
    high = high.rename(columns=colmap(columns, 'high'))
    lim_df = pd.concat([q1, q3, iqr, low, high], axis=1)

    # Add columns.
    func = _get_outlier_cols_func(columns, lim_df)
    out_df = df.apply(func, axis=1, result_type='expand')
    df = pd.concat([df, out_df], axis=1)
    return df

def load_region_summary(
    dataset: str,
    blacklist: bool = False,
    regions: types.PatientRegions = 'all') -> None:
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'region-summary-{hash}.csv')
    df = pd.read_csv(filepath)
    if blacklist:
        # Exclude blacklisted records.
        filepath = os.path.join(set.path, 'region-blacklist.csv')
        black_df = pd.read_csv(filepath)
        df = df.merge(black_df, how='left', on=['patient', 'region'], indicator=True)
        df = df[df['_merge'] == 'left_only']
        df = df.drop(columns='_merge')
    return df

def get_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    cols = {
        'patient-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Load values.
        patient = set.patient(pat)
        size = patient.ct_size()
        spacing = patient.ct_spacing()

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'patient-id': pat,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = df.append(data, ignore_index=True)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get summary.
    df = get_ct_summary(dataset, regions=regions)

    # Save summary.
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{hash}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{hash}.csv')
    return pd.read_csv(filepath)

def create_region_figures(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    # Get regions.
    if type(regions) == str:
        if regions == 'all':
            regions = list(set.list_regions().region.unique())
        else:
            regions = [regions]

    # Keep regions with patients.
    region_df = load_region_summary(dataset, regions=regions)
    regions = list(sorted(region_df.region.unique()))

    # Add 'extent-mm' outlier info.
    columns = ['extent-mm-x', 'extent-mm-y', 'extent-mm-z']
    region_df = add_region_summary_outliers(region_df, columns)

    # Set PDF margins.
    table_1_t_padding = 25
    table_1_l_padding = 5
    table_2_t_padding = 55
    table_2_l_padding = 5
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    logging.info(f"Creating region figures for dataset '{dataset}', regions '{regions}'...")
    for region in tqdm(regions):
        # Create PDF.
        pdf = FPDF()
        pdf.set_section_title_styles(
            TitleStyle(
                font_family='Times',
                font_style='B',
                font_size_pt=24,
                color=0,
                t_margin=5,
                l_margin=15,
                b_margin=0
            )
        ) 

        for pat in tqdm(pats, leave=False):
            # Skip if patient doesn't have region.
            patient = set.patient(pat)
            if not patient.has_region(region):
                continue

            # Add patient/region title.
            pdf.add_page()
            pdf.start_section(pat)

            # Save table 1.
            num_cols = 2
            cell_height = 2 * pdf.font_size
            cell_width = (img_width - 2 * table_1_l_padding) / num_cols
            table_1_data = [('Connected', 'Connected Prop.')]
            reg_record = region_df[(region_df['patient'] == pat) & (region_df['region'] == region)].iloc[0]
            table_1_data.append((reg_record.connected, f"{reg_record['connected-p']:.2f}"))
            for i, row in enumerate(table_1_data):
                if i == 0:
                    pdf.set_font('Helvetica', 'B', 12)
                else:
                    pdf.set_font('Helvetica', '', 12)
                pdf.set_xy(img_l_margin + table_1_l_padding, img_t_margin + table_1_t_padding + i * cell_height)
                for value in row:
                    pdf.cell(cell_width, cell_height, str(value), border=1)

            # Save table 2.
            num_cols = 4
            cell_width = (img_width - 2 * table_2_l_padding) / num_cols
            table_2_data = [('Axis', 'Outlier', 'Extent', 'Num. IQR')]
            axes = ['x', 'y', 'z']
            for axis, column in zip(axes, columns):
                table_2_data.append((axis, reg_record[f'{column}-out-dir'], f'{reg_record[column]:.2f}', f"{reg_record[f'{column}-out-iqr']:.2f}"))
            for i, row in enumerate(table_2_data):
                if i == 0:
                    pdf.set_font('Helvetica', 'B', 12)
                else:
                    pdf.set_font('Helvetica', '', 12)
                pdf.set_xy(img_l_margin + table_2_l_padding, img_t_margin + table_2_t_padding + i * cell_height)
                for value in row:
                    pdf.cell(cell_width, cell_height, str(value), border=1)

            # Save images.
            views = ['axial', 'coronal', 'sagittal']
            img_coords = ((img_l_margin + img_width, img_t_margin), (img_l_margin, img_t_margin + img_height), (img_l_margin + img_width, img_t_margin + img_height))
            for view, page_coord in zip(views, img_coords):
                # Set figure.
                plot_patient_regions(dataset, pat, centre_of=region, colours=['y'], crop=region, extent=True, regions=region, view=view, window=(3000, 500))

                # Save temp file.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plt.savefig(filepath)
                plt.close()

                # Add image to report.
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)

        # Save PDF.
        filepath = os.path.join(set.path, 'reports', 'region-figures', f'{region}.pdf') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.output(filepath, 'F')

def _hash_regions(regions: types.PatientRegions) -> str:
    return hashlib.sha1(json.dumps(regions).encode('utf-8')).hexdigest()
