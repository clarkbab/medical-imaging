import inspect
import numpy as np
from skimage.measure import label   

def largest_cc_3D(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.bool_:
        raise ValueError(f"'{inspect[0][3]}' expected a boolean array, got '{a.dtype}'.")

    # Check that there are some foreground pixels.
    labels = label(a)
    if labels.max() == 0:
        return np.zeros_like(a)
    
    # Calculate largest component.
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_cc

def largest_cc_4D(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.bool_:
        raise ValueError(f"'{inspect[0][3]}' expected a boolean array, got '{a.dtype}'.")
    ccs = []
    for data in a:
        cc = largest_cc_3D(data)
        ccs.append(cc)
    output = np.stack(ccs, axis=0)
    return output

def largest_cc_5D(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.bool_:
        raise ValueError(f"'{inspect[0][3]}' expected a boolean array, got '{a.dtype}'.")
    ccs = []
    for data in a:
        cc = largest_cc_4D(data)
        ccs.append(cc)
    output = np.stack(ccs, axis=0)
    return output
        