from enum import Enum

class RegionList(list, Enum):
    # MICCAI 2015 dataset (PDDCA).
    MICCAI = ['Bone_Mandible', 'Brainstem', 'Glnd_Submand_L', 'Glnd_Submand_R', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R']

    # PMCC vendor evaluation.
    PMCC_COMP = [
        'A_Aorta', 'A_Pulmonary', 'Bladder', 'Bone_Ilium_L', 'Bone_Ilium_R', 'Bone_Mandible', 'BrachialPlex_L',
        'BrachialPlex_R', 'Brain', 'Brainstem', 'Bronchus', 'Breast_L', 'Breast_R', 'Cavity_Oral', 'Chestwall',
        'Cochlea_L', 'Cochlea_R', 'Colon_Sigmoid', 'Duodenum', 'Esophagus', 'Eye_L', 'Eye_R', 'Femur_Head_L',
        'Femur_Head_R', 'Gallbladder', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Glottis', 'Heart', 'Kidney_L',
        'Kidney_R', 'Larynx', 'Lens_L', 'Lens_R', 'Lips', 'Liver', 'Lung_L', 'Lung_R', 'OpticChiasm',
        'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 'Pericardium', 'Prostate', 'Rectum', 'Skin',
        'SpinalCanal', 'SpinalCord', 'Spleen', 'Stomach', 'Trachea'
    ]

    # Replan dataset.
    REPLAN = []

    # Transfer learning project.
    TL = ['BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R']
