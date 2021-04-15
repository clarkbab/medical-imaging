import numpy as np
from skimage.measure import label   

def largest_connected_component(a):
    """
    returns: a volume with the largest connected component only.
    args:
        a: a binary prediction or label.
    """
    # Check that there's at least 1 connected component.
    labels = label(a)
    assert( labels.max() != 0 )
    
    # Calculate largest component.
    comp = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return comp
