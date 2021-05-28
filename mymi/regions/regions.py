import matplotlib.pyplot as plt
from enum import Enum

class Regions(Enum):
    BrachialPlexus_L = 0
    BrachialPlexus_R = 1
    Brain = 2
    BrainStem = 3
    Cochlea_L = 4
    Cochlea_R = 5
    MedullaOblongata = 6
    Neck_L = 7
    Neck_R = 8
    OpticChiasm = 9
    OpticNerve_L = 10
    OpticNerve_R = 11
    OralCavity = 12
    Parotid_L = 13
    Parotid_R = 14
    SpinalCord = 15
    Submandibular_L = 16
    Submandibular_R = 17
    Thyroid = 18

# Define region color map.
palette = plt.cm.tab20
class RegionColours:
    BrachialPlexus_L = palette(0)
    BrachialPlexus_R = palette(1)
    Brain = palette(12)
    BrainStem = palette(13)
    Cochlea_L = palette(2)
    Cochlea_R = palette(3)
    MedullaOblongata = palette(14)
    Neck_L = palette(4)
    Neck_R = palette(5)
    OpticChiasm = palette(15)
    OpticNerve_L = palette(6)
    OpticNerve_R = palette(7)
    OralCavity = palette(16)
    Parotid_L = palette(8)
    Parotid_R = palette(9)
    SpinalCord = palette(17)
    Submandibular_L = palette(10)
    Submandibular_R = palette(11)
    Thyroid = palette(18)
