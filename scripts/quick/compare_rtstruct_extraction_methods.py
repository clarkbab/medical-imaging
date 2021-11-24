import os
from os.path import dirname as up
import pandas as pd
import pathlib
import sys
from tqdm import tqdm

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(filepath)))
sys.path.append(mymi_dir)
from mymi import config
from mymi import dataset
from mymi import logging
from mymi.metrics import dice, distances

num_pats = 100

# Load ID maps.
pmcc_map_path = 'S:\ImageStore\HN_AI_Contourer\DICOM_Data\datasets\dicom\PMCC-HN-TRAIN-OLD\\anon-nifti-map.csv'
pmcc_map_df = pd.read_csv(pmcc_map_path, names=['anon-id', 'patient-id'], header=0)
price_map_path = 'S:\ImageStore\HN_AI_Contourer\DICOM_Data\match-hn_structures_v2.csv'
price_map_df = pd.read_csv(price_map_path, index_col=0)

# Perform inner join.
map_df = pmcc_map_df.merge(price_map_df, left_on='patient-id', right_on='UR', how='inner')[['anon-id', 'patient-id', 'Series_UID']]

# Get datasets.
pmcc_set = dataset.get('PMCC-HN-TRAIN-OLD', 'nifti')
price_set = dataset.get('PMCC-HN-TRAIN-PRICE', 'nifti')

cols = {
    'patient-id': str,
    'pmcc-id': str,
    'price-id': str,
    'region': str,
    'metric': str,
    'value': float
}
df = pd.DataFrame(columns=cols.keys())

pats = list(map_df['patient-id'].unique())
for pat in tqdm(pats[:num_pats]):
    # Get mapped IDs.
    row = map_df[map_df['patient-id'] == pat].iloc[0]
    anon_id = row['anon-id']
    series_uid = row.Series_UID
    
    # Get spacing.
    pmcc_pat = pmcc_set.patient(anon_id)
    pmcc_spacing = pmcc_pat.ct_spacing()
    price_pat = price_set.patient(series_uid)
    price_spacing = price_pat.ct_spacing()
    if pmcc_spacing != price_spacing:
        logging.error(f"Spacing not consistent '{pmcc_spacing}' and '{price_spacing}', for patient '{pmcc_pat}' and '{price_pat}'.")
        continue
    
    # Get labels.
    pmcc_labels = pmcc_pat.region_data()
    price_labels = price_pat.region_data()
    
    # Evaluate labels.
    for region in pmcc_labels.keys():
        # Get region label.
        pmcc_label = pmcc_labels[region]
        if price_pat.has_region(region):
            price_label = price_labels[region]
        else:
            logging.error(f"Region '{region}' not found for patient '{price_pat}'.")
            continue
        
        data = {
            'patient-id': pat,
            'region': region,
            'pmcc-id': anon_id,
            'price-id': series_uid
        }
        
        # Get DSC.
        data['metric'] = 'dice'
        data['value'] = dice(pmcc_label, price_label)
        df = df.append(data, ignore_index=True)
        
        # Distances.
        dists = distances(pmcc_label, price_label, pmcc_spacing)
        for metric, value in dists.items():
            data['metric'] = metric
            data['value'] = value
            df = df.append(data, ignore_index=True)

# Set types.
df = df.astype(cols)

# Save evaluation.
filepath = os.path.join(config.directories.files, 'rtstruct-extract-comp.csv')
df.to_csv(filepath, index=False)
