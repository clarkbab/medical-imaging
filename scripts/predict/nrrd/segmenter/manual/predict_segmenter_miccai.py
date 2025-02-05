import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.predictions.datasets.nrrd import create_multi_segmenter_predictions

reses = ['112', '222']
n_regionses = [1, 2, 4, 8, 9]
regionses = [
    [
        'Bone_Mandible',     # 0
        'Brainstem',         # 1
        'Glnd_Submand_L',    # 2
        'Glnd_Submand_R',    # 3
        'Parotid_L',         # 7
        'Parotid_R'         # 8
    ],
    [
        ['Bone_Mandible','Brainstem'],         # 0
        ['Glnd_Submand_L','Glnd_Submand_R'],   # 1
        ['OpticChiasm','Bone_Mandible'],       # 2
        ['OpticNrv_L','OpticNrv_R'],           # 3
        ['Parotid_L','Parotid_R']             # 4
    ],
    [
        ['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R'],
        ['OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R'],
        ['OpticChiasm','Bone_Mandible','Brainstem','Glnd_Submand_L']
    ],
    [
        ['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R'],
        ['OpticChiasm','Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L']
    ],
    [
        ['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']
    ]
]
short_regionses = [
    [
        'BM',
        'BS',
        'SL',
        'SR',
        'PL',
        'PR'
    ],
    [
        'BM_BS',
        'SL_SR',
        'OC',
        'OL_OR',
        'PL_PR'
    ],
    [
        'BM_BS_SL_SR',
        'OL_OR_PL_PR',
        'OC'
    ],
    [
        'BM_BS_SL_SR_OL_OR_PL_PR',
        'OC'
    ],
    [
        'ALL'
    ]
]

for n_regions, regions, short_regions in zip(n_regionses, regionses, short_regionses):
    for region, short_region in zip(regions, short_regions):
        for res in reses:
            dataset = f'MICCAI-2015-{res}'
            region_word = 'region' if n_regions == 1 else 'regions'
            model = ('segmenter-miccai-no-background', f'{n_regions}-{region_word}-{short_region}-{res}', 'best')
            create_multi_segmenter_predictions(dataset, region, model, check_epochs=False, use_loader_split_file=True)
