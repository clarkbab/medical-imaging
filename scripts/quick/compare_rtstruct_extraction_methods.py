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

# Get datasets.
mymi_set = dataset.get('EXTRACT-MYMI', 'nifti')
plastimatch_set = dataset.get('EXTRACT-PLASTIMATCH', 'nifti')

cols = {
    'patient-id': str,
    'region': str,
    'metric': str,
    'value': float
}
df = pd.DataFrame(columns=cols.keys())

pats = mymi_set.list_patients()
for pat in tqdm(pats):
    # Get spacing.
    mymi_pat = mymi_set.patient(pat)
    mymi_spacing = mymi_pat.ct_spacing()
    plastimatch_pat = plastimatch_set.patient(pat)
    plastimatch_spacing = plastimatch_pat.ct_spacing()
    if mymi_spacing != plastimatch_spacing:
        logging.error(f"Spacing not consistent '{mymi_spacing}' and '{plastimatch_spacing}', for patient '{mymi_pat}' and '{plastimatch_pat}'.")
        continue
    
    # Get labels.
    mymi_labels = mymi_pat.region_data()
    plastimatch_labels = plastimatch_pat.region_data()
    
    # Evaluate labels.
    for region in mymi_labels.keys():
        # Get region label.
        mymi_label = mymi_labels[region]
        if plastimatch_pat.has_region(region):
            plastimatch_label = plastimatch_labels[region]
        else:
            logging.error(f"Region '{region}' not found for patient '{plastimatch_pat}'.")
            continue
        
        data = {
            'patient-id': pat,
            'region': region,
        }
        
        # Get DSC.
        data['metric'] = 'dice'
        data['value'] = dice(mymi_label, plastimatch_label)
        df = df.append(data, ignore_index=True)
        
        # Distances.
        dists = distances(mymi_label, plastimatch_label, mymi_spacing)
        for metric, value in dists.items():
            data['metric'] = metric
            data['value'] = value
            df = df.append(data, ignore_index=True)

# Set types.
df = df.astype(cols)

# Save evaluation.
filepath = os.path.join(config.directories.files, 'rtstruct-extract-comp.csv')
df.to_csv(filepath, index=False)
