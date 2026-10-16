from dicomset.utils import load_nifti, one_hot_encode, save_nifti
from tqdm.auto import tqdm

pat_ids = [1, 2, 3, 4]
# Only patient 1 has manual markers at the moment.
pat_ids = [1]

for p in pat_ids:
    # Copy manual segmentations from Slicer -> Dicomset.
    for phase in tqdm(range(10)):
        s = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\Patient{p:02d}\markers\phase_{phase}\markers-lps.nii"
        d, a = load_nifti(s)
        m = one_hot_encode(d, background=False, dim=3)
        for i, mi in enumerate(m):
            d = fr"D:\Brett\data\mymi\datasets\nifti\VALKIM\data\patients\PAT{p}\study_0\regions\series_{phase}\bc_Marker_{i + 1}.nii.gz"
            save_nifti(mi, a, d)
