from enum import Enum

# RegionNames = [
#     'Bone_Mandible',
#     'BrachialPlex_L',
#     'BrachialPlex_R',
#     'Brain',
#     'Brainstem',
#     'Cavity_Oral',
#     'Cochlea_L',
#     'Cochlea_R',
#     'Esophagus_S',
#     'Eye_L',
#     'Eye_R',
#     'Glnd_Submand_L',
#     'Glnd_Submand_R',
#     'Glottis',
#     'Larynx',
#     'Lens_L',
#     'Lens_R',
#     'Musc_Constrict',
#     'OpticChiasm',
#     'OpticNrv_L',
#     'OpticNrv_R',
#     'Parotid_L',
#     'Parotid_R',
#     'SpinalCord',
# ]

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
