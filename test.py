from mymi.processing.dataset.nifti import convert_to_training

dataset = 'PMCC-HN-TRAIN'
resolutions = [(4, 4, 4), (2, 2, 2), (1, 1, 2)]
short_resolutions = ['444', '222', '112']

for res, short_res in zip(resolutions, short_resolutions):
    dest_dataset = f'{dataset}-{short_res}'
    convert_to_training(dataset, dest_dataset=dest_dataset, output_spacing=res)