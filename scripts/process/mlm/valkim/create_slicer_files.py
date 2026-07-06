from dicomset.dicom.utils import from_ct_dicom, from_rtstruct_dicom, list_rtstruct_regions
from dicomset.utils import plot_volume, save_nifti, change_orientation, load_nrrd
import numpy as np
import os
from tqdm import tqdm

pat_id = "Patient01"

for phase in tqdm(range(10)):
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\2{phase + 1:02d}-iDose {phase * 10}% Linear, iDose (4)"
    data, affine = from_ct_dicom(f)
    f = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\243-RT LUNG Plan\RS.000000.dcm"
    regions = ['Fiducial_1', 'Fiducial_2', 'Fiducial_3']
    regions, regions_data = from_rtstruct_dicom(f, data.shape, affine, l=None, r=regions)
    ras_affine = change_orientation(affine, 'LPS', 'RAS')
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\ct.nii"
    os.makedirs(os.path.dirname(f), exist_ok=True)
    save_nifti(data, ras_affine, f)
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\PlanningFiles\{pat_id}\markers\phase_{phase}\avg_markers.nii"
    # Add background and convert to integer label map.
    regions_data = np.concatenate([np.zeros(regions_data.shape[1:])[np.newaxis, ...], regions_data], axis=0)
    regions_data = np.argmax(regions_data, axis=0).astype(np.int16)
    save_nifti(regions_data, ras_affine, f)
