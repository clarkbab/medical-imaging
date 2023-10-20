# from mymi.processing.dataset.nifti import convert_to_training

# dataset = 'PMCC-HN-TRAIN'
# resolutions = [(4, 4, 4), (2, 2, 2), (1, 1, 2)]
# short_resolutions = ['444', '222', '112']

# for res, short_res in zip(resolutions, short_resolutions):
#     dest_dataset = f'{dataset}-{short_res}'
#     convert_to_training(dataset, dest_dataset=dest_dataset, output_spacing=res)

# from mymi.models import replace_ckpt_alias
# model = ('segmenter-miccai-arch-modification', 'channels-1-seed-42-cw-1-ivw-1', 'last')
# print(replace_ckpt_alias(model))

# import torch

# from mymi.multi_class.gradcam import create_heatmap

# dataset = 'MICCAI-2015'
# pat_id = '0522c0001'
# model = ('segmenter-miccai-numbers', '1-region-BM-112-seed-42', 'best')
# model_region = 'Bone_Mandible'
# model_spacing = (1, 1, 2)
# region = 'Bone_Mandible'
# layers = ['5', '12', '19', '26', '33']
# device = torch.device('cuda:0')

# create_heatmap(dataset, pat_id, model, model_region, model_spacing, region, layers, device=device)

from mymi.processing.dataset.nifti import convert_to_training
from mymi.regions import RegionList

dataset = 'PMCC-HN-REPLAN'
output_spacing = (1.171875, 1.171875, 2.0)

convert_to_training(dataset, output_spacing=output_spacing)

