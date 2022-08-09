
# Tolerances in mm.
class RegionTolerances:
    BrachialPlexus_L = None
    BrachialPlexus_R = None
    Brain = 1.01
    BrainStem = 2.5
    Cochlea_L = 1.25
    Cochlea_R = 1.25
    Lens_L = 0.98
    Lens_R = 0.98
    Mandible = 1.01
    OpticNerve_L = 2.5
    OpticNerve_R = 2.5
    OralCavity = None
    Parotid_L = 2.85
    Parotid_R = 2.85
    SpinalCord = 2.93
    Submandibular_L = 2.02
    Submandibular_R = 2.02

def get_region_tolerance(region):
    return getattr(RegionTolerances, region)
