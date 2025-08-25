from fpdf import FPDF, TitleStyle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.ndimage.measurements import label as label_objects
from tqdm import tqdm
from typing import *
from uuid import uuid1

from mymi import config
from mymi.datasets import TrainingDataset
from mymi.geometry import fov, fov_centre
from mymi import logging
from mymi.processing import get_object, one_hot_encode
from mymi.typing import *
from mymi.utils import *

def create_ct_figures_report(
    dataset: str,
    region: Optional[Regions] = None) -> None:
    logging.arg_log('Creating CT figures (TrainingDataset)', ('dataset', 'region'), (dataset, region))

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

    # Get patients.
    set = TrainingDataset(dataset)
    sample_ids = set.list_samples(region=region)

    for sample_id in sample_ids:
        # Load input.
        input = set.sample(sample_id).input

        # Show images.
        pdf.add_page()
        pdf.start_section(f'Sample: {sample_id}')

        # Save images.
        axes = list(range(3))
        img_coords = (
            (img_l_margin, img_t_margin),
            (img_l_margin + img_width, img_t_margin),
            (img_l_margin, img_t_margin + img_height)
        )
        for axis, page_coord in zip(axes, img_coords):
            # Save figure.
            z = input.shape[axis] // 2
            filepath = os.path.join(config.directories.temp, f'{uuid1().hex}.png')
            plot_patients(dataset, sample_id, savepath=filepath, z=z, view=axis)
            plt.close()

            # Add image to report.
            pdf.image(filepath, *page_coord, w=img_width, h=img_height)

            # Delete temp file.
            os.remove(filepath)

    # Save PDF.
    filepath = os.path.join(set.path, 'reports', 'ct-figures.pdf') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath, 'F')

def create_ct_summary(dataset: str) -> None:
    # Get summary.
    df = get_ct_summary(dataset)

    # Save summary.
    set = TrainingDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def get_ct_summary(dataset: str) -> pd.DataFrame:
    logging.info(f"Creating CT summary for dataset '{dataset}'.")

    # Get patients.
    set = TrainingDataset(dataset)
    sample_ids = set.list_samples()

    cols = {
        'dataset': str,
        'sample-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for sample_id in tqdm(sample_ids):
        # Load values.
        sample = set.sample(sample_id)
        size = sample.size
        spacing = sample.spacing

        # Calculate FOV.
        fov = sample.fov

        for axis in range(len(size)):
            data = {
                'dataset': dataset,
                'sample-id': sample_id,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_counts_report(
    dataset: str,
    region: Optional[Regions] = None) -> None:
    # Get regions.
    set = TrainingDataset(dataset)
    regions = set.list_regions() if region is None else arg_to_list(region, str)

    # Generate counts report.
    df = get_region_counts(dataset, region=regions)
    filepath = os.path.join(set.path, 'reports', 'region-counts', encode(regions), 'region-counts.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_region_figures(
    dataset: str,
    regions: Regions = 'all') -> None:
    # Get dataset.
    set = TrainingDataset(dataset)

    # Get regions.
    if type(regions) == str:
        if regions == 'all':
            regions = list(sorted(set.list_regions().region.unique()))
        else:
            regions = [regions]

    # Filter regions that don't exist in dataset.
    pat_regions = list(sorted(set.list_regions().region.unique()))
    regions = [r for r in pat_regions if r in regions]

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

        # Define partitions.
        partitions = ['train', 'validation', 'test']

        for partition in tqdm(partitions, leave=False):
            # Load samples.
            part = set.partition(partition)
            samples = part.list_samples(regions=region)
            if len(samples) == 0:
                continue

            # Start partition section.
            pdf.add_page()
            pdf.start_section(f'Partition: {partition}')

            for s in tqdm(samples, leave=False):
                # Load sample.
                sample = part.sample(s)

                # Start info section.
                pdf.add_page()
                pdf.start_section(f'Sample: {s}', level=1)
                pdf.start_section('Info', level=2)

                # Add table.
                table_t_margin = 50
                table_l_margin = 12
                table_cols = 5
                table_line_height = 2 * pdf.font_size
                table_col_widths = (15, 35, 30, 45, 45)
                table_width = 180
                table_data = [('ID', 'Volume [vox]', 'Volume [p]', 'Extent Centre [vox]', 'Extent Width [vox]')]
                obj_df = get_object_summary(dataset, partition, s, region)
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
                    pdf.start_section(f'Object: {i}', level=2)

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
                        plot_sample_regions(dataset, partition, s, centre=region, colours=['y'], postproc=postproc, regions=region, show_extent=True, view=view, window=(3000, 500))

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

def get_region_counts(
    dataset: str,
    region: Optional[Regions] = None) -> pd.DataFrame:
    # Get regions.
    set = TrainingDataset(dataset)
    regions = set.list_regions() if region is None else arg_to_list(region, str)

    cols = {
        'dataset': str,
        'region': str,
        'n-samples': int
    }
    df = pd.DataFrame(columns=cols.keys())
    for region in regions:
        n_samples = len(set.list_samples(region=region))
        data = {
            'dataset': dataset,
            'region': region,
            'n-samples': n_samples
        }
        df = append_row(df, data)

    df = df.astype(cols)

    return df

def get_object_summary(
    dataset: str,
    partition: str,
    sample: str,
    region: str) -> pd.DataFrame:
    set = TrainingDatset(dataset)
    samp = set.partition(partition).sample(sample)
    spacing = eval(set.params.spacing[0])
    label = samp.label(regions=region)[region]
    objs, n_objs = label_objects(label, structure=np.ones((3, 3, 3)))
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
    for i in range(n_objs):
        obj = objs[:, :, :, i]
        data = {}

        # Get extent.
        min, max = extent(obj)
        width = tuple(np.array(max) - min)
        data['extent-width-vox'] = str(width)
        
        # Get centre of extent.
        extent_centre = fov_centre(obj)
        data['extent-centre-vox'] = str(extent_centre)

        # Add volume.
        vox_volume = spacing[0] * spacing[1] * spacing[2]
        n_voxels = obj.sum()
        volume = n_voxels * vox_volume
        data['volume-vox'] = n_voxels
        data['volume-p'] = n_voxels / tot_voxels
        data['volume-mm3'] = volume

        df = df.append(data, ignore_index=True)

    df = df.astype(cols)
    return df

def load_ct_summary(dataset: str) -> pd.DataFrame:
    set = TrainingDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"CT summary doesn't exist for dataset '{dataset}'.")
    return pd.read_csv(filepath)

def load_region_counts_report(
    dataset: str,
    region: Optional[Regions] = None) -> pd.DataFrame:
    # Get regions.
    set = TrainingDataset(dataset)
    regions = set.list_regions() if region is None else arg_to_list(region, str)

    # Generate counts report.
    filepath = os.path.join(set.path, 'reports', 'region-counts', encode(regions), 'region-counts.csv')
    df = pd.read_csv(filepath)

    return df
