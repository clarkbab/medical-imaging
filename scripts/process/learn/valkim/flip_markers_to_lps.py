# Slicer operates in RAS+ coordinates, we prefer LPS+.
from dicomset.utils import change_orientation, load_nifti, save_nifti
from tqdm import tqdm

pat_id = "Patient01"

for phase in tqdm(range(10)):
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\ct.nii"
    data, affine = load_nifti(f)
    lps_affine = change_orientation(affine, 'RAS', 'LPS')
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\ct-lps.nii"
    save_nifti(data, lps_affine, f)
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\markers.nii"
    data, affine = load_nifti(f)
    lps_affine = change_orientation(affine, 'RAS', 'LPS')
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\markers-lps.nii"
    save_nifti(data, lps_affine, f)
