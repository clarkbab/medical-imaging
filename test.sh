. scripts/utilities/bash_functions.sh

REGIONS=(
    'Bone_Mandible'     # 0
    'BrachialPlex_L'    # 1
    'BrachialPlex_R'    # 2
    'Brain'             # 3
    'Brainstem'         # 4
    'Cavity_Oral'       # 5
    'Cochlea_L'         # 6
    'Cochlea_R'         # 7
    'Esophagus_S'       # 8
    'Eye_L'             # 9
    'Eye_R'             # 10
    'Glnd_Submand_L'    # 11
    'Glnd_Submand_R'    # 12
    'Glottis'           # 13
    'GTVp'              # 14
    'Larynx'            # 15
    'Lens_L'            # 16
    'Lens_R'            # 17
    'Musc_Constrict'    # 18
    'OpticChiasm'       # 19
    'OpticNrv_L'        # 20
    'OpticNrv_R'        # 21
    'Parotid_L'         # 22
    'Parotid_R'         # 23
    'SpinalCord'        # 24
)
REGION=$(join "${REGIONS[@]}")
echo $REGION