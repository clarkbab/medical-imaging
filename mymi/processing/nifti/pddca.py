from mymi.datasets import NiftiDataset
from mymi.datasets.nifti import recreate as recreate_nifti
from mymi.datasets.nifti.utils import create_ct, create_region
from mymi.geometry import foreground_fov
from mymi.regions import regions_to_list
from mymi.transforms import crop
from mymi.utils import *

def create_pddca_cropped_dataset(
    dry_run: bool = True,
    recreate: bool = False) -> None:
    boundary_min = ['ts_Parotid_R', 'ts_Bone_Mandible', ['ts_Glnd_Submand_L', 'ts_Glnd_Submand_R']]
    boundary_max = ['ts_Parotid_L', 'ts_Brainstem', ['ts_OpticNrv_L', 'ts_OpticNrv_R']]
    margin_mm = 30
    region_ids = regions_to_list('rl:pddca')

    set = NiftiDataset('PDDCA')
    if recreate:
        dest_set = recreate_nifti('PDDCA-PP', dry_run=dry_run)
    else:
        dest_set = NiftiDataset('PDDCA-PP')
    pat_ids = set.list_patients()
    for p in pat_ids:
        pat = set.patient(p)
        ct_data = pat.ct_data
        spacing = pat.ct_spacing
        offset = pat.ct_offset

        # Get crop coords from boundary organs.
        min_mm = []
        for i, r in enumerate(boundary_min):
            r = arg_to_list(r, RegionID)
            region_data = pat.region_data(r)
            label = np.zeros_like(ct_data)
            for k, v in region_data.items():
                label = np.logical_or(label, v)
            
            # Get structure fov.
            fov_l = foreground_fov(label, spacing=spacing, offset=offset, use_patient_coords=True)
            min_mm.append(fov_l[0][i])

        # Add margin.
        min_mm = np.array(min_mm) - margin_mm
        min_mm = tuple(min_mm)

        max_mm = []
        for i, r in enumerate(boundary_max):
            r = arg_to_list(r, RegionID)
            region_data = pat.region_data(r)
            label = np.zeros_like(ct_data)
            for k, v in region_data.items():
                label = np.logical_or(label, v)
            
            # Get structure fov.
            fov_l = foreground_fov(label, spacing=spacing, offset=offset, use_patient_coords=True)
            max_mm.append(fov_l[1][i])

        # Add margin.
        max_mm = np.array(max_mm) + margin_mm
        max_mm = tuple(max_mm)

        # Crop and save CT image.
        crop_mm = (min_mm, max_mm)
        ct_data = crop(ct_data, crop_mm, spacing=spacing, offset=offset, use_patient_coords=True)
        create_ct(dest_set.id, p, 'study_0', 'series_0', ct_data, spacing, offset, dry_run=dry_run)

        # Crop non-totalseg labels.
        for r in region_ids:
            if not pat.has_region(r):
                continue
            rdata = pat.region_data(r)[r]
            rdata = crop(rdata, crop_mm, spacing=spacing, offset=offset, use_patient_coords=True)
            create_region(dest_set.id, p, 'study_0', 'series_0', r, rdata, spacing, offset, dry_run=dry_run)
    