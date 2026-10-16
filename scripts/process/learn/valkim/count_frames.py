import os
import pandas as pd
from tqdm.auto import tqdm

from mymi.utils.args import arg_to_list
from mymi.utils.cdog import list_raw_arcs, list_tiff, list_tiff_fractions

pat_ids = [1, 2]

fractions = 'all'
# fractions = [1]
arcs = 'all'
# arcs = [1]
image_path = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

rows = []
for p in tqdm(pat_ids, desc='Patients'):
    pat_id = f'Patient{p:02d}'
    pat_path = os.path.join(image_path, pat_id)
    pat_fractions = arg_to_list(fractions, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    for fx in tqdm(pat_fractions, desc='Fractions', leave=False):
        fraction_path = os.path.join(pat_path, f"Fx{fx:02d}")
        fx_arcs = arg_to_list(arcs, int, literals={'all': lambda: list_raw_arcs(fraction_path)})
        for arc in tqdm(fx_arcs, desc='Arcs', leave=False):
            n_frames = len(list_tiff(fraction_path, arc))
            rows.append({'patient': pat_id, 'fraction': fx, 'arc': arc, 'n-frames': n_frames})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nTotal frames: {df['n-frames'].sum()}")
