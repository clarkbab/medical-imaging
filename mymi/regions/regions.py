from enum import Enum
import hashlib
import json

from mymi import types

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

def hash_regions(regions: types.PatientRegions) -> str:
    return hashlib.sha1(json.dumps(regions).encode('utf-8')).hexdigest()
