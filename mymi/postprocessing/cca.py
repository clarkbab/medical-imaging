import numpy as np
from skimage.measure import label   

def largest_connected_component(a):
    """
    returns: a 3D array with largest CC only.
    args:
        a: a 3D array with binary values.
    """
    # Check that there's at least 1 connected component.
    labels = label(a)
    assert( labels.max() != 0 )
    
    # Calculate largest component.
    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    return largest_cc

def batch_largest_connected_component(a):
    """
    returns: a batch of 3D arrays with largest CC only.
    args:
        a: a batch of 3D arrays with binary values.
    """
    a_new = np.ndarray()
    for i in range(len(a)):
        # Get the largest connected component.
        largest_cc = largest_connected_component(a[i])
        a_new[i] = largest_cc

    return a
        