import matplotlib.pyplot as plt
from enum import Enum

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
    MedullaOblongata = 9
    Neck_L = 10
    Neck_R = 11
    OpticChiasm = 12
    OpticNerve_L = 13
    OpticNerve_R = 14
    OralCavity = 15
    Parotid_L = 16
    Parotid_R = 17
    SpinalCord = 18
    Submandibular_L = 19
    Submandibular_R = 20
    Thyroid = 21

# Define region color map.
palette_tab20 = plt.cm.tab20
palette_tab20b = plt.cm.tab20b
class RegionColours:
    BrachialPlexus_L = palette_tab20(0)
    BrachialPlexus_R = palette_tab20(1)
    Brain = palette_tab20(14)
    BrainStem = palette_tab20(15)
    Cochlea_L = palette_tab20(10)
    Cochlea_R = palette_tab20(11)
    Lens_L = palette_tab20(6)
    Lens_R = palette_tab20(7)
    Mandible = palette_tab20(16)
    MedullaOblongata = palette_tab20(17)
    Neck_L = palette_tab20(2)
    Neck_R = palette_tab20(3)
    OpticChiasm = palette_tab20(18)
    OpticNerve_L = palette_tab20(8)
    OpticNerve_R = palette_tab20(9)
    OralCavity = palette_tab20(19)
    Parotid_L = palette_tab20(4)
    Parotid_R = palette_tab20(5)
    SpinalCord = palette_tab20b(0)
    Submandibular_L = palette_tab20(12)
    Submandibular_R = palette_tab20(13)
    Thyroid = palette_tab20b(1)
