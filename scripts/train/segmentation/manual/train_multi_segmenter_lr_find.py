import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.training import train_multi_segmenter

# Definitions.
model = 'segmenter-miccai-lr-find'
resolutions = ['112', '222', '444']
precisions = [32, 'bf16']
precisions = [16]
seeds = [42, 43, 44, 45, 46]

# regions = ['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']
# for seed in seeds:
#     for precision in precisions:
#         for resolution in resolutions:
#             dataset = f'MICCAI-2015-{resolution}'
#             run = f'9-regions-{resolution}-{precision}-{seed}' 
#             train_multi_segmenter(dataset, regions, model, run, lr_find=True, n_workers=8, precision=precision, random_seed=seed, use_loader_split_file=True)

regions = ['Bone_Mandible', 'Brainstem', 'Glnd_Submand_L', 'OpticChiasm']
short_regions = ['BM', 'BS', 'SL', 'OC']
for seed in seeds:
    for precision in precisions:
        for resolution in resolutions:
            for region, short_region in zip(regions, short_regions):
                dataset = f'MICCAI-2015-{resolution}'
                run = f'1-region-{short_region}-{precision}-{seed}'
                train_multi_segmenter(dataset, region, model, run, lr_find=True, n_workers=8, precision=precision, random_seed=seed, use_loader_split_file=True)
