import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.training import train_multi_segmenter

# Definitions.
model = 'segmenter-miccai-lr-find'
resolutions = ['112', '222', '444']
precisions = ['bf16']
regions = ['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']

for precision in precisions:
    for resolution in resolutions:
        dataset = f'MICCAI-2015-{resolution}'
        run = f'9-regions-{resolution}-{precision}' 
        train_multi_segmenter(dataset, regions, model, run, lr_find=True, n_workers=8, precision=precision, use_loader_split_file=True)
