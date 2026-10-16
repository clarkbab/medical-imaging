import os
from tqdm.auto import tqdm

from mymi.utils.cdog import process_tiff_patient

# Patient01: filename angle is kv-source, scale is IEC61217.
# Patient02: filename angle is kv-source, scale is IEC61217.
# Patient03: ?
# Patient04: filename angle is mv-source, scale is IEC61217.

# pat_ids = [1, 2, 3, 4]
# filename_angles = ['kv-source', 'kv-source', 'kv-source', 'mv-source']
# pat_ids = [2]
# filename_angles = ['kv-source']
pat_ids = [3]
filename_angles = ['kv-source']

fractions = 'all'
arcs = 'all'
image_path = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

for i, p in tqdm(enumerate(pat_ids), desc='Patients'):
    pat_id = f'Patient{p:02d}'
    pat_path = os.path.join(image_path, pat_id)
    process_tiff_patient(pat_path, fraction=fractions, arc=arcs, filename_angle=filename_angles[i], save_data=True)
