from mymi.processing.nifti import convert_to_nnunet_single_region

dataset = 'PDDCA-PP'
first_dataset = 0
region = parse_arg('region', str)
print(region)
spacing = (0.75, 0.75, 1.25)
convert_to_nnunet_single_region(
    dataset,
    first_dataset,
    region,
    spacing=spacing,
)
