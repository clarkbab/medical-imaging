import hashlib
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List

from mymi import dataset as ds
from mymi.postprocessing import get_extent
from mymi import types

def get_region_summary(
    dataset: str,
    regions: List[str]) -> pd.DataFrame:
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=regions)

    cols = {
        'patient': str,
        'region': str,
        'axis': str,
        'extent-mm': float,
        'spacing': float
    }
    df = pd.DataFrame(columns=cols.keys())

    axes = [0, 1, 2]

    # Initialise empty data structure.
    data = {}
    for region in regions:
        data[region] = {}
        for axis in axes:
            data[region][axis] = []

    for pat in tqdm(pats):
        # Get spacing.
        spacing = set.patient(pat).ct_spacing()

        # Get region data.
        pat_regions = set.patient(pat).list_regions(whitelist=regions)
        rs_data = set.patient(pat).region_data(regions=pat_regions)

        # Add extents for all regions.
        for r in rs_data.keys():
            r_data = rs_data[r]
            min, max = get_extent(r_data)
            for axis in axes:
                extent_vox = max[axis] - min[axis]
                extent_mm = extent_vox * spacing[axis]
                df = df.append({
                    'patient': pat,
                    'region': r,
                    'axis': axis,
                    'extent-mm': extent_mm,
                    'spacing': spacing[axis]
                }, ignore_index=True)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_summary(
    dataset: str,
    regions: List[str]) -> None:
    # Generate counts report.
    df = get_region_summary(dataset, regions)

    # Save report.
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'region-summary-{hash}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_region_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    set = ds.get(dataset, 'nifti')
    hash = _hash_regions(regions)
    filepath = os.path.join(set.path, 'reports', f'region-summary-{hash}.csv')
    return pd.read_csv(filepath)

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

        for axis in range(3):
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
    pat_ids: types.PatientIDs = 'all',
    regions: types.PatientRegions = 'all',
    report_name: str = None) -> None:
    """
    effect: Generates a PDF report of dataset segmentations.
    args:
        dataset: the dataset name.
    kwargs:
        pat_ids: the patients to show.
        regions: the regions to show.
        report_name: the name of the report.
    """
    # Get patients.
    ds = DicomDataset(dataset)
    pats = ds.list_patients(regions=regions)

    # Filter patients.
    pats = list(filter(filterOnPatIDs(pat_ids), pats))
    pats = list(filter(ds.filterOnRegions(regions), pats))
    pats = 

    # Get regions.
    if regions == 'all':
        regions = ds.list_regions().region.unique() 
    elif isinstance(regions, str):
        regions = [regions]

    # Create PDF.
    report = FPDF()
    report.set_font('Arial', 'B', 16)

    for region in tqdm(regions):
        for pat in tqdm(pats, leave=False):
            # Skip if patient doesn't have region.
            if not ds.patient(pat).has_region(region):
                continue

            # Add patient/region title.
            report.add_page()
            text = f"Region: {region}, Patient: {pat}"
            report.cell(0, 0, text, ln=1)

            # Get region centroid.
            summary = ds.patient(pat).region_summary(clear_cache=clear_cache, regions=region).iloc[0].to_dict()
            centroid = (int(summary['centroid-voxels-x']), int(summary['centroid-voxels-y']), int(summary['centroid-voxels-z']))

            # Save orthogonal plots.
            views = ['sagittal', 'coronal', 'axial']
            origins = ((0, 20), (100, 20), (0, 120))
            for c, o, v in zip(centroid, origins, views):
                # Set figure.
                plotter.plot_patient_regions(pat, c, regions=region, show=False, view=v)

                # Save temp file.
                filename = f"patient-{pat}-region-{region}-view-{v}.png"
                filepath = os.path.join(config.directories.temp, filename)
                plt.savefig(filepath)

                # Add image to report.
                report.image(filepath, *o, w=100, h=100)

    # Save PDF.
    if report_name:
        filename = report_name
    else:
        filename = f"report-{dataset}.pdf"
    filepath = os.path.join(config.directories.files, filename) 
    report.output(filepath, 'F')

def _hash_regions(regions: types.PatientRegions) -> str:
    return hashlib.sha1(json.dumps(regions).encode('utf-8')).hexdigest()
