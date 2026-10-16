import os

from mymi.utils.cdog import join_tiff_arcs

# Join Fx05, arcs [1, 2] of Patient04 into a single arc, saved as arc 1.
pat_id = 2
fraction = 6
arcs = [3, 4, 5]

image_path = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

pat_id = f'Patient{pat_id:02d}'
pat_path = os.path.join(image_path, pat_id)
fraction_path = os.path.join(pat_path, f'Fx{fraction:02d}')

join_tiff_arcs(fraction_path, arcs, remove_sources=True)
