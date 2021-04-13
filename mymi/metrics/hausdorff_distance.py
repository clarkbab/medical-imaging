import numpy as np
from scipy.spatial.distance import directed_hausdorff

def batch_hausdorff_distance(pred, label, distance='euclidean', spacing=(1.0, 1.0, 1.0)):
    """
    returns: the mean Hausdorff distance across the batch.
    args:
        pred: the batch of binary network predictions, e.g. shape (n, 512, 512, 212).
        label: the batch of labels, e.g. shape (n, 512, 512, 212)
    kwargs:
        distance: the distance metric to use.
        spacing: the voxel spacing used.
    """
    assert pred.shape == label.shape

    # Calculate Hausdorff distance for each item in batch.
    hd_dists = []
    for i in range(len(pred)):
        # Get item from batch.
        p, l = pred[i], label[i]

        # Get symmetric Hausdorff distance.
        hd_dist_a = directed_hausdorff_distance(p, l, distance=distance, spacing=spacing)
        hd_dist_b = directed_hausdorff_distance(l, p, distance=distance, spacing=spacing)
        hd_dist = max(hd_dist_a, hd_dist_b)
        hd_dists.append(hd_dist)

    return np.mean(hd_dists)

def directed_hausdorff_distance(a, b, distance='euclidean', spacing=(1.0, 1.0, 1.0)):
    """
    returns: the directed Hausdorff distance from volumes a to b.
    args:
        a: the first volume.
        b: the second volume.
    kwargs:
        distance: the distance metric to use.
        spacing: the spacing between voxels.
    """
    # Get coordinates of non-zero voxels.
    a_coords, b_coords = np.argwhere(a != 0), np.argwhere(b != 0)
    
    # Store the max distance (max) from a voxel in p to the closest point in l.
    max_min_dist = None
    
    for a_i in a_coords:
        # Convert to true spacing.
        a_true_i = a_i * spacing
        
        # Store the minimum distance from a_true to any voxel in b.
        min_dist = None
        
        for b_j in b_coords:
            # Convert to true spacing.
            b_true_j = b_j * spacing
            
            # Find the distance between a_i and b_j.
            if distance == 'euclidean':
                dist = euclidean_distance(a_true_i, b_true_j)

            # Update the minimum distance if necessary.
            if min_dist is None or dist < min_dist:
                min_dist = dist

        # Update the maximum minimum distance if necessary.
        if max_min_dist is None or min_dist > max_min_dist:
            max_min_dist = min_dist

        # Reset the minimum distance.
        min_dist = None
        
    return max_min_dist

def euclidean_distance(a, b):
    """
    returns: the Euclidean distance between 3-dimensional points a and b.
    args:
        a: the first 3D point.
        b: the second 3D point.
    """
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
