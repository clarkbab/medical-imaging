from enum import Enum
import os
import pandas as pd

from mymi import config
from mymi.typing import *
from mymi.utils import *

EXPANDED_REGION_MAP = {
    'Lungs': ['Lung_Lower_Lobe_L', 'Lung_Lower_Lobe_R', 'Lung_Middle_Lobe_R', 'Lung_Upper_Lobe_L', 'Lung_Upper_Lobe_R'],
    'Ribs': ['Rib_L_1', 'Rib_L_2', 'Rib_L_3', 'Rib_L_4', 'Rib_L_5', 'Rib_L_6', 'Rib_L_7', 'Rib_L_8', 'Rib_L_9', 'Rib_L_10', 'Rib_L_11', 'Rib_L_12', 'Rib_R_1', 'Rib_R_2', 'Rib_R_3', 'Rib_R_4', 'Rib_R_5', 'Rib_R_6', 'Rib_R_7', 'Rib_R_8', 'Rib_R_9', 'Rib_R_10', 'Rib_R_11', 'Rib_R_12'],
    'Vertebrae': ['Vertebrae_C1', 'Vertebrae_C2', 'Vertebrae_C3', 'Vertebrae_C4', 'Vertebrae_C5', 'Vertebrae_C6', 'Vertebrae_C7', 'Vertebrae_L1', 'Vertebrae_L2', 'Vertebrae_L3', 'Vertebrae_L4', 'Vertebrae_L5', 'Vertebrae_T1', 'Vertebrae_T2', 'Vertebrae_T3', 'Vertebrae_T4', 'Vertebrae_T5', 'Vertebrae_T6', 'Vertebrae_T7', 'Vertebrae_T8', 'Vertebrae_T9', 'Vertebrae_T10', 'Vertebrae_T11', 'Vertebrae_T12', 'Vertebrae_S1'],
}

class RegionList(list, Enum):
    # HaN-Seg dataset.
    HAN_SEG = [
        'A_Carotid_L',
        'A_Carotid_R',
        'Arytenoid',
        'Bone_Mandible',
        'Brainstem',
        'BuccalMucosa',
        'Cavity_Oral',
        'Cochlea_L',
        'Cochlea_R',
        'Cricopharyngeus',
        'Esophagus_S',
        'Eye_AL',
        'Eye_AR',
        'Eye_PL',
        'Eye_PR',
        'Glnd_Lacrimal_L',
        'Glnd_Lacrimal_R',
        'Glnd_Submand_L',
        'Glnd_Submand_R',
        'Glnd_Thyroid',
        'Glottis',
        'Larynx_SG',
        'Lips',
        'OpticChiasm',
        'OpticNrv_L',
        'OpticNrv_R',
        'Parotid_L',
        'Parotid_R',
        'Pituitary',
        'SpinalCord'
    ]
    HAN_SEG_SHORT = [
        'Car_L',
        'Car_R',
        'Aryt',
        'Mand',
        'Brstem',
        'BucMuc',
        'OralC',
        'Coch_L',
        'Coch_R',
        'Crico',
        'Eso_S',
        'Eye_AL',
        'Eye_AR',
        'Eye_PL',
        'Eye_PR',
        'Lacr_L',
        'Lacr_R',
        'Subm_L',
        'Subm_R',
        'Thyr',
        'Glot',
        'Laryn_SG',
        'Lips',
        'OptChi',
        'OptNrv_L',
        'OptNrv_R',
        'Par_L',
        'Par_R',
        'Pit',
        'SpinCord'
    ]

    # MICCAI 2015 dataset (PDDCA).
    MICCAI = ['Bone_Mandible', 'Brainstem', 'Glnd_Submand_L', 'Glnd_Submand_R', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R']
    MICCAI_CVG_THRESHOLDS = [0.71, 0.66, 0.54, 0.54, 0.25, 0.40, 0.44, 0.63, 0.62]
    MICCAI_INVERSE_VOLUMES = [
        1.81605095e-05,
        3.75567497e-05,
        1.35979999e-04,
        1.34588032e-04,
        1.71684281e-03,
        1.44678695e-03,
        1.63991258e-03,
        3.45656440e-05,
        3.38292316e-05
    ]
    MICCAI_SHORT = ['BM', 'BS', 'SL', 'SR', 'OC', 'OL', 'OR', 'PL', 'PR']
    assert len(MICCAI) == len(MICCAI_CVG_THRESHOLDS) == len(MICCAI_INVERSE_VOLUMES) == len(MICCAI_SHORT)

    # PMCC vendor evaluation.
    PMCC_COMP = [ 'A_Aorta', 'A_Pulmonary', 'Bladder', 'Bone_Ilium_L', 'Bone_Ilium_R', 'Bone_Mandible', 'BrachialPlex_L',
        'BrachialPlex_R', 'Brain', 'Brainstem', 'Bronchus', 'Breast_L', 'Breast_R', 'Cavity_Oral', 'Chestwall',
        'Cochlea_L', 'Cochlea_R', 'Colon_Sigmoid', 'Duodenum', 'Esophagus', 'Eye_L', 'Eye_R', 'Femur_Head_L',
        'Femur_Head_R', 'Gallbladder', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Glottis', 'Heart', 'Kidney_L',
        'Kidney_R', 'Larynx', 'Lens_L', 'Lens_R', 'Lips', 'Liver', 'Lung_L', 'Lung_R', 'OpticChiasm',
        'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 'Pericardium', 'Prostate', 'Rectum', 'Skin',
        'SpinalCanal', 'SpinalCord', 'Spleen', 'Stomach', 'Trachea'
    ]

    # REPLAN dataset.
    PMCC_REPLAN_ALL = [
        'Bone_Mandible',
        'BrachialPlex_L',
        'BrachialPlex_R',
        'Brain',
        'Brainstem',
        'Cavity_Oral',
        'Cochlea_L',
        'Cochlea_R',
        'Esophagus_S',
        'Eye_L',
        'Eye_R',
        'GTVp',
        'Glnd_Submand_L',
        'Glnd_Submand_R',
        'Glottis',
        'Larynx',
        'Lens_L',
        'Lens_R',
        'Musc_Constrict',
        'OpticChiasm',
        'OpticNrv_L',
        'OpticNrv_R',
        'Parotid_L',
        'Parotid_R',
        'SpinalCord'
    ]
    PMCC_REPLAN_ALL_CVG_THRESHOLDS = [0.05] * len(PMCC_REPLAN_ALL)
    PMCC_REPLAN_ALL_INVERSE_VOLUMES = [
        1.2937933022269647e-05,
        0.00011431883233820832,
        0.0001160421372817061,
        7.04443295518197e-07,
        3.783638005916425e-05,
        8.45901601256752e-06,
        0.003001606135764661,
        0.0033476280849826525,
        8.306122217507318e-05,
        0.00010363227956173657,
        0.00010157727789858287,
        2.9514039194735845e-05,
        9.421716755555447e-05,
        9.506018822834804e-05,
        4.3696848314141504e-05,
        2.33866869029497e-05,
        0.0031775138796888763,
        0.0031586113628843137,
        4.2703493087156644e-05,
        0.0011668649939908305,
        0.0007894397655065801,
        0.000797525580829336,
        3.106137352694446e-05,
        2.9621564096699957e-05,
        3.592121091349627e-05
    ]
    PMCC_REPLAN_ALL_SHORT = [
        'BM',
        'BL',
        'BR',
        'B',
        'BS',
        'CO',
        'CL',
        'CR',
        'E',
        'EL',
        'ER',
        'GTV',
        'SL',
        'SR',
        'G',
        'L',
        'LL',
        'LR',
        'MC',
        'OC',
        'OL',
        'OR',
        'PL',
        'PR',
        'SC'
    ]
    assert len(PMCC_REPLAN_ALL) == len(PMCC_REPLAN_ALL_INVERSE_VOLUMES) == len(PMCC_REPLAN_ALL_SHORT)

    # REPLAN dataset (short).
    PMCC_REPLAN = [
        'Bone_Mandible',
        'BrachialPlex_L',
        'BrachialPlex_R',
        'Brain',
        'Brainstem',
        'Cavity_Oral',
        'Esophagus_S',
        'GTVp',
        'Glnd_Submand_L',
        'Glnd_Submand_R',
        'Larynx',
        'Lens_L',
        'Lens_R',
        'Musc_Constrict',
        'Parotid_L',
        'Parotid_R',
        'SpinalCord'
    ]
    PMCC_REPLAN_CVG_THRESHOLDS = [0.05] * len(PMCC_REPLAN)
    # PMCC_REPLAN_CVG_THRESHOLDS = [
    #     0.71,
    #     0.25,
    #     0.25,
    #     0.78,
    #     0.63,
    #     0.63,
    #     0.55,
    #     0.05,
    #     0.57,
    #     0.58,
    #     0.2,
    #     0.36,
    #     0.37,
    #     0.2,
    #     0.62,
    #     0.63,
    #     0.53
    # ]
    PMCC_REPLAN_INVERSE_VOLUMES = [
        1.2937933022269647e-05,
        0.00011431883233820832,
        0.0001160421372817061,
        7.04443295518197e-07,
        3.783638005916425e-05,
        8.45901601256752e-06,
        8.306122217507318e-05,
        2.9514039194735845e-05,
        9.421716755555447e-05,
        9.506018822834804e-05,
        2.33866869029497e-05,
        0.0031775138796888763,
        0.0031586113628843137,
        4.2703493087156644e-05,
        3.106137352694446e-05,
        2.9621564096699957e-05,
        3.592121091349627e-05
    ]
    PMCC_REPLAN_SHORT = [
        'BM',
        'BL',
        'BR',
        'B',
        'BS',
        'CO',
        'E',
        'GTV',
        'SL',
        'SR',
        'L',
        'LL',
        'LR',
        'MC',
        'PL',
        'PR',
        'SC'
    ]
    assert len(PMCC_REPLAN) == len(PMCC_REPLAN_INVERSE_VOLUMES) == len(PMCC_REPLAN_SHORT)

    # REPLAN dataset.
    PMCC_REPLAN_EYES = [
        'Eye_L',
        'Eye_R',
        'Lens_L',
        'Lens_R',
    ]
    PMCC_REPLAN_EYES_CVG_THRESHOLDS = [
        0.5,
        0.5,
        0.36,
        0.37,
    ]
    PMCC_REPLAN_EYES_INVERSE_VOLUMES = [
        0.00010363227956173657,
        0.00010157727789858287,
        0.0031775138796888763,
        0.0031586113628843137,
    ]
    PMCC_REPLAN_EYES_SHORT = [
        'EL',
        'ER',
        'LL',
        'LR',
    ]
    assert len(PMCC_REPLAN_EYES) == len(PMCC_REPLAN_EYES_INVERSE_VOLUMES) == len(PMCC_REPLAN_EYES_SHORT)

    # Transfer learning project.
    PMCC = ['BrachialPlexus_L', 'BrachialPlexus_R', 'Brain', 'BrainStem', 'Cochlea_L', 'Cochlea_R', 'Lens_L', 'Lens_R', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'OralCavity', 'Parotid_L', 'Parotid_R', 'SpinalCord', 'Submandibular_L', 'Submandibular_R']
    PMCC_CVG_THRESHOLDS = [0.31, 0.35, 0.78, 0.63, 0.27, 0.32, 0.36, 0.37, 0.71, 0.4, 0.42, 0.63, 0.62, 0.63, 0.53, 0.57, 0.58]
    PMCC_INVERSE_VOLUMES = [
        0.00011376029489088039,
        0.00011060966775471057,
        7.446522874686845e-07,
        3.964170420857003e-05,
        0.002744113436394272,
        0.0030073116457818953,
        0.002957342943345055,
        0.002988210189604558,
        1.3595619210428649e-05,
        0.000977472289486429,
        0.000962569199385967,
        9.200393030692723e-06,
        3.0352277602361224e-05,
        3.072187177824903e-05,
        3.976200542837415e-05,
        0.0001100993441549776,
        0.00010420091523896904
    ]
    PMCC_SHORT = ['BL', 'BR', 'B', 'BS', 'CL', 'CR', 'LL', 'LR', 'M', 'OL', 'OR', 'OC', 'PL', 'PR', 'SC', 'SL', 'SR']
    assert len(PMCC) == len(PMCC_CVG_THRESHOLDS) == len(PMCC_INVERSE_VOLUMES) == len(PMCC_SHORT)

# Behaves like 'arg_to_list', but also handles special 'RL:<region list>' format.
def regions_to_list(regions: RegionIDs, **kwargs) -> RegionIDs:
    if regions is None:
        return None

    regions = arg_to_list(regions, RegionID, **kwargs)
    rs = []
    for r in regions:
        if r.startswith('rl:'): 
            # Expand str to list of regions.
            rl_name = r.split(':')[-1]
            if hasattr(RegionList, rl_name):
                r = list(getattr(RegionList, rl_name))
            else:
                filepath = os.path.join(config.directories.config, 'region-lists', f'{rl_name}.csv')
                if not os.path.exists(filepath):
                    raise ValueError(f"Region list '{rl_name}' not found. Filepath: {filepath}")
                df = pd.read_csv(filepath, header=None)
                r = list(sorted(df[0]))
            rs += r
        else:
            rs.append(r)

    return rs
    