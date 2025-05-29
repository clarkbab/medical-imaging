from mymi.utils import is_windows

from .segmenter_grad_norm import *
if not is_windows():
    from .segmenter_pcgrad import *
from .segmenter_uncertainty_weighting import *
