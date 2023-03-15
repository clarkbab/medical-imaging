from enum import Enum

MODE = 0

if MODE == 0:
    RegionNames = [
        'Bone_Mandible',        # 0
        'BrachialPlex_L',       # 1
        'BrachialPlex_R',       # 2
        'Brain',                # 3
        'Brainstem',            # 4
        'Cavity_Oral',          # 5
        'Cochlea_L',            # 6
        'Cochlea_R',            # 7
        'Esophagus_S',          # 8
        'Eye_L',                # 9
        'Eye_R',                # 10
        'Glnd_Submand_L',       # 11
        'Glnd_Submand_R',       # 12
        'Glottis',              # 13
        'GTVp',                 # 14
        'Larynx',               # 15
        'Lens_L',               # 16
        'Lens_R',               # 17
        'Musc_Constrict',       # 18
        'OpticChiasm',          # 19
        'OpticNrv_L',           # 20
        'OpticNrv_R',           # 21
        'Parotid_L',            # 22
        'Parotid_R',            # 23
        'SpinalCord',           # 24
    ]
elif MODE == 1:
    class Regions(Enum):
        BrachialPlexus_L = 0
        BrachialPlexus_R = 1
        Brain = 2
        BrainStem = 3
        Cochlea_L = 4
        Cochlea_R = 5
        Lens_L = 6
        Lens_R = 7
        Mandible = 8
        OpticNerve_L = 9
        OpticNerve_R = 10
        OralCavity = 11
        Parotid_L = 12
        Parotid_R = 13
        SpinalCord = 14
        Submandibular_L = 15
        Submandibular_R = 16

    RegionNames = [r.name for r in Regions]

# For multi-class training - 'background' is channel '0'.
RegionChannelMap = dict((r, i + 1) for i, r in enumerate(RegionNames))
RegionChannelMap['Background'] = 0
ChannelRegionMap = dict((c, r) for r, c in RegionChannelMap.items())
