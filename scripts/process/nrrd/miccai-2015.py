import os
import shutil
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import datasets as ds
from mymi import logging

STRUCT_MAP = {
    'BrainStem': 'Brainstem',
    'Chiasm': 'OpticChiasm',
    'Mandible': 'Bone_Mandible',
    'OpticNerve_L': 'OpticNrv_L',
    'OpticNerve_R': 'OpticNrv_R',
    'Submandibular_L': 'Glnd_Submand_L',
    'Submandibular_R': 'Glnd_Submand_R',
}

# Get patient IDS.
set = ds.get('MICCAI-2015', 'nrrd')
pat_ids = os.listdir(set.path)

# Create data folder.
data_path = os.path.join(set.path, 'data')
os.makedirs(data_path, exist_ok=True)

# Move data for each patient.
logging.info(f'Moving MICCAI-2015 data for {len(pat_ids)} patients.')
for pat_id in tqdm(pat_ids):
    if pat_id == 'data':    # Allow for multiple script runs.
        continue

    # Move CT file.
    pat_path = os.path.join(set.path, pat_id)
    ct_path = os.path.join(pat_path, 'img.nrrd') 
    if not os.path.exists(ct_path):
        raise ValueError(f"No CT image found for patient '{pat_id}' at path '{ct_path}'.")
    dest_ct_path = os.path.join(data_path, 'ct', f'{pat_id}.nrrd')
    os.makedirs(os.path.dirname(dest_ct_path), exist_ok=True)
    shutil.copyfile(ct_path, dest_ct_path)

    # Move regions.
    structs_path = os.path.join(pat_path, 'structures')
    structs = os.listdir(structs_path)
    for struct in structs:
        struct_path = os.path.join(structs_path, struct) 
        struct_name = struct.replace('.nrrd', '')
        # Map from MICCAI-2015 structures names to our internal names.
        if struct_name in STRUCT_MAP:
            struct_name = STRUCT_MAP[struct_name]
        dest_struct_path = os.path.join(data_path, 'regions', struct_name, f'{pat_id}.nrrd')
        os.makedirs(os.path.dirname(dest_struct_path), exist_ok=True)
        shutil.copyfile(struct_path, dest_struct_path)

    # Remove original patient path.
    shutil.rmtree(pat_path)
