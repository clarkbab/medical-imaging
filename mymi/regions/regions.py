
RegionNames = [
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
    'Glnd_Submand_L',
    'Glnd_Submand_R',
    'Glottis',
    'GTVp',
    'Larynx',
    'Lens_L',
    'Lens_R',
    'Musc_Constrict',
    'OpticChiasm',
    'OpticNrv_L',
    'OpticNrv_R',
    'Parotid_L',
    'Parotid_R',
    'SpinalCord',
]

# For multi-class training - 'background' is channel '0'.
RegionChannelMap = dict((r, i + 1) for i, r in enumerate(RegionNames))
RegionChannelMap['Background'] = 0
ChannelRegionMap = dict((c, r) for r, c in RegionChannelMap.items())
