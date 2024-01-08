from mymi.processing.dataset.nifti import convert_adaptive_brain_crop_to_training_v2
from mymi.regions import RegionList

dataset = 'PMCC-HN-REPLAN'
spacing = (2, 2, 2)
dest_dataset = 'PMCC-HN-REPLAN-ADPT-222'
crop_mm = (330, 380, 500)
regions = RegionList.PMCC_REPLAN

convert_adaptive_brain_crop_to_training_v2(dataset, dest_dataset=dest_dataset, crop_mm=crop_mm, spacing=spacing, region=regions)

# from mymi.processing.dataset.nifti import convert_population_to_training_v2
# from mymi.regions import RegionList

# dataset = 'PMCC-HN-REPLAN'
# # output_spacing = (1.171875, 1.171875, 2)
# output_spacing = (2, 2, 2)
# dest_dataset = 'PMCC-HN-REPLAN-POP-222'
# output_size_mm = (250, 400, 500)
# regions = RegionList.PMCC_REPLAN
# use_compression = True

# convert_population_to_training_v2(dataset, dest_dataset=dest_dataset, output_size_mm=output_size_mm, output_spacing=output_spacing, region=regions, use_compression=use_compression)

# from mymi.processing.dataset.nifti import convert_adaptive_mirror_to_training_v2

# dataset = 'PMCC-HN-REPLAN'
# output_spacing = (2, 2, 2)
# dest_dataset = 'PMCC-HN-REPLAN-ADPTM-222'
# output_size_mm = (250, 400, 500)
# regions = 'RL:PMCC_REPLAN'
# use_compression = True

# convert_adaptive_mirror_to_training_v2(dataset, dest_dataset=dest_dataset, output_size_mm=output_size_mm, output_spacing=output_spacing, region=regions, use_compression=use_compression)
