from typing import Optional

# Tolerances in mm.
class RegionTolerances:
    Bone_Mandible = 1.01
    BrachialPlex_L = 1          # Not defined in Nikolov et al. 
    BrachialPlex_R = 1          # Not defined in Nikolov et al.
    Brain = 1.01
    Brainstem = 2.5
    Cavity_Oral = 1             # Not defined in Nikolov et al.
    Cochlea_L = 1.25
    Cochlea_R = 1.25
    Esophagus_S = 1             # Not defined in Nikolov et al. 
    Eye_L = 1                   # Not defined in Nikolov et al. 
    Eye_R = 1                   # Not defined in Nikolov et al. 
    Glnd_Submand_L = 2.02
    Glnd_Submand_R = 2.02
    Glottis = 1                 # Not defined in Nikolov et al. 
    GTVp = 1                    # Not defined in Nikolov et al. 
    Larynx = 1                  # Not defined in Nikolov et al. 
    Lens_L = 0.98
    Lens_R = 0.98
    Musc_Constrict = 1          # Not defined in Nikolov et al. 
    OpticChiasm = 1             # Not defined in Nikolov et al. 
    OpticNrv_L = 2.5
    OpticNrv_R = 2.5
    Parotid_L = 2.85
    Parotid_R = 2.85
    SpinalCord = 2.93

def get_region_tolerance(region: str) -> Optional[float]:
    if hasattr(RegionTolerances, region):
        return getattr(RegionTolerances, region)
    else:
        return None
