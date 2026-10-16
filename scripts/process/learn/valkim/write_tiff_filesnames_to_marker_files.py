from dicomset.utils import load_csv, save_csv
import os

a = 3
f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient01\Fx01\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
df = load_csv(f)

f = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx01\kV"
tiff_files = [f for f in os.listdir(f) if f.endswith('.tiff')]
arc_files = list(sorted([f for f in tiff_files if f.startswith(f'Ch1_{a}')]))
assert len(arc_files) == len(df)

df['Filename'] = arc_files

f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation\Patient01\Fx01\Arc{a:02d}\MarkerLocationsGA_CouchShift_adapted_edited.txt"
save_csv(df, f)
