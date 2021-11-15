from mymi import types

class RegionPatchSizes:
    BrachialPlexus_L = (170, 165, 80)
    BrachialPlexus_R = (170, 165, 80)
    Brain = (170, 205, 90)
    BrainStem = (105, 100, 65)
    Cochlea_L = (70, 65, 30)
    Cochlea_R = (70, 65, 30)
    Lens_L = (55, 50, 25)
    Lens_R = (55, 50, 25)
    Mandible = (180, 165, 80)
    OpticNerve_L = (75, 90, 35)
    OpticNerve_R = (75, 90, 35)
    OralCavity = (180, 170, 75)
    Parotid_L = (120, 125, 80)
    Parotid_R = (120, 125, 80)
    SpinalCord = (85, 210, 155)
    Submandibular_L = (80, 85, 55)
    Submandibular_R = (80, 85, 55)

def get_patch_size(region: str) -> types.ImageSize3D:
    return getattr(RegionPatchSizes, region)
