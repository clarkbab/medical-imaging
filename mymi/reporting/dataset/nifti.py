from fpdf import FPDF, TitleStyle
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.ndimage.measurements import label as label_objects
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple
from uuid import uuid1

from mymi import config
from mymi import dataset as ds
from mymi.evaluation.dataset.nifti import load_localiser_evaluation, load_segmenter_evaluation
from mymi.geometry import get_extent, get_extent_centre
from mymi import logging
from mymi.plotter.dataset.nifti import plot_patient_localiser_prediction, plot_patient_regions, plot_patient_segmenter_prediction
from mymi.postprocessing import get_largest_cc, get_object, one_hot_encode
from mymi.regions import hash_regions
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
        'connected-extent-mm-x': float,
        'connected-extent-mm-y': float,
        'connected-extent-mm-z': float,
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
    hash = hash_regions(regions)
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
    regions: List[str],
    blacklist: bool = False) -> None:
    set = ds.get(dataset, 'nifti')
    hash = hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'region-summary-{hash}.csv')
    df = pd.read_csv(filepath)

    # Filter blacklisted records.
    if blacklist:
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
    hash = hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{hash}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    set = ds.get(dataset, 'nifti')
    hash = hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{hash}.csv')
    return pd.read_csv(filepath)

def create_region_figures(
    dataset: str,
    regions: List[str]) -> None:
    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    # Get regions.
    if type(regions) == str:
        if regions == 'all':
            regions = list(sorted(set.list_regions().region.unique()))
        else:
            regions = [regions]

    # Keep regions with patients.
    region_df = load_region_summary(dataset, regions)
    regions = list(sorted(region_df.region.unique()))

    # Add 'extent-mm' outlier info.
    columns = ['extent-mm-x', 'extent-mm-y', 'extent-mm-z']
    region_df = add_region_summary_outliers(region_df, columns)

    # Set PDF margins.
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
            table_cols = 5
            table_line_height = 2 * pdf.font_size
            table_col_widths = (15, 35, 30, 45, 45)
            table_width = 180
            table_data = [('ID', 'Volume [vox]', 'Volume [p]', 'Extent Centre [vox]', 'Extent Width [vox]')]
            obj_df = get_object_summary(dataset, pat, region)
            for i, row in obj_df.iterrows():
                table_data.append((
                    str(i),
                    str(row['volume-vox']),
                    f"{row['volume-p']:.3f}",
                    row['extent-centre-vox'],
                    row['extent-width-vox']
                ))
            for i, row in enumerate(table_data):
                if i == 0:
                    pdf.set_font('Helvetica', 'B', 12)
                else:
                    pdf.set_font('Helvetica', '', 12)
                pdf.set_xy(table_l_margin, table_t_margin + i * table_line_height)
                for j, value in enumerate(row):
                    pdf.cell(table_col_widths[j], table_line_height, value, border=1)

            for i, row in obj_df.iterrows():
                # Start object section.
                pdf.add_page()
                pdf.start_section(f'Object: {i}', level=1)

                # Save images.
                views = ['axial', 'coronal', 'sagittal']
                img_coords = (
                    (img_l_margin, img_t_margin),
                    (img_l_margin + img_width, img_t_margin),
                    (img_l_margin, img_t_margin + img_height)
                )
                for view, page_coord in zip(views, img_coords):
                    # Set figure.
                    def postproc(a: np.ndarray):
                        return get_object(a, i)
                    plot_patient_regions(dataset, pat, centre_of=region, colours=['y'], postproc=postproc, regions=region, show_extent=True, view=view, window=(3000, 500))

                    # Save temp file.
                    filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                    plt.savefig(filepath)
                    plt.close()

                    # Add image to report.
                    pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                    # Delete temp file.
                    os.remove(filepath)

        # Save PDF.
        filepath = os.path.join(set.path, 'reports', 'region-figures', f'{region}.pdf') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.output(filepath, 'F')

def get_object_summary(
    dataset: str,
    patient: str,
    region: str) -> pd.DataFrame:
    pat = ds.get(dataset, 'nifti').patient(patient)
    spacing = pat.ct_spacing()
    label = pat.region_data(regions=region)[region]
    objs, num_objs = label_objects(label, structure=np.ones((3, 3, 3)))
    objs = one_hot_encode(objs)
    
    cols = {
        'extent-centre-vox': str,
        'extent-width-vox': str,
        'volume-mm3': float,
        'volume-p': float,
        'volume-vox': int
    }
    df = pd.DataFrame(columns=cols.keys())
    
    tot_voxels = label.sum()
    for i in range(num_objs):
        obj = objs[:, :, :, i]
        data = {}

        # Get extent.
        min, max = get_extent(obj)
        width = tuple(np.array(max) - min)
        data['extent-width-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = get_extent_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        num_voxels = obj.sum()
        volume = num_voxels * vox_volume
        data['volume-vox'] = num_voxels
        data['volume-p'] = num_voxels / tot_voxels
        data['volume-mm3'] = volume

        df = df.append(data, ignore_index=True)

    df = df.astype(cols)
    return df

def create_localiser_figures(
    dataset: str,
    regions: List[str],
    localisers: List[Tuple[str, str, str]]) -> None:
    assert len(regions) == len(localisers)

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    # Filter regions that don't exist in dataset.
    pat_regions = list(sorted(set.list_regions().region.unique()))
    regions = [r for r in pat_regions if r in regions]

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    logging.info(f"Creating localiser figures for dataset '{dataset}', regions '{regions}'...")
    for region, localiser in tqdm(list(zip(regions, localisers))):
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
        table_cols = 5
        table_line_height = 2 * pdf.font_size
        table_col_widths = (40, 40, 40)
        table_width = 180
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
            table_cols = 5
            table_line_height = 2 * pdf.font_size
            table_col_widths = (40, 40)
            table_width = 180
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
            views = ['axial', 'coronal', 'sagittal']
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Set figure.
                plot_patient_localiser_prediction(dataset, pat, region, localiser, centre_of=region, colour='y', show_patch=True, view=view, window=(3000, 500))

                # Save temp file.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plt.savefig(filepath)
                plt.close()

                # Add image to report.
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                # Delete temp file.
                os.remove(filepath)

        # Save PDF.
        filepath = os.path.join(set.path, 'reports', 'localiser-figures', f'{region}.pdf') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.output(filepath, 'F')

def create_segmenter_figures(
    dataset: str,
    regions: List[str],
    segmenters: List[Tuple[str, str, str]]) -> None:
    assert len(regions) == len(segmenters)

    # Get patients.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    # Filter regions that don't exist in dataset.
    pat_regions = list(sorted(set.list_regions().region.unique()))
    regions = [r for r in pat_regions if r in regions]

    # Set PDF margins.
    img_t_margin = 30
    img_l_margin = 5
    img_width = 100
    img_height = 100

    logging.info(f"Creating segmenter figures for dataset '{dataset}', regions '{regions}'...")
    for region, segmenter in tqdm(list(zip(regions, segmenters))):
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
        eval_df = load_segmenter_evaluation(dataset, segmenter)

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
            views = ['axial', 'coronal', 'sagittal']
            img_coords = (
                (img_l_margin, img_t_margin),
                (img_l_margin + img_width, img_t_margin),
                (img_l_margin, img_t_margin + img_height)
            )
            for view, page_coord in zip(views, img_coords):
                # Set figure.
                plot_patient_segmenter_prediction(dataset, pat, region, segmenter, centre_of=region, view=view, window=(3000, 500))

                # Save temp file.
                filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
                plt.savefig(filepath)
                plt.close()

                # Add image to report.
                pdf.image(filepath, *page_coord, w=img_width, h=img_height)

                # Delete temp file.
                os.remove(filepath)

        # Save PDF.
        filepath = os.path.join(set.path, 'reports', 'segmenter-figures', f'{region}.pdf') 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pdf.output(filepath, 'F')
