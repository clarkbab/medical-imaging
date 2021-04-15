import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch

def hausdorff_distance(a, b, distance='euclidean', spacing=(1., 1., 1.)):
    """
    returns: the Hausdorff distance between pred and label.
    args:
        a: a 3D array of binary values.
        b: a 3D array of binary values.
    kwargs:
        distance: the distance metric to use.
        spacing: the voxel spacing used.
    """
    assert a.shape == b.shape

    # Get symmetric Hausdorff distance.
    dist_a = directed_hausdorff_distance(a, b, distance=distance, spacing=spacing)
    dist_b = directed_hausdorff_distance(b, a, distance=distance, spacing=spacing)
    hd_dist = max(dist_a, dist_b)

    return hd_dist

def batch_hausdorff_distance(a, b, distance='euclidean', spacing=(1.0, 1.0, 1.0)):
    """
    returns: the mean Hausdorff distance across the batch.
    args:
        a: a batch of 3D arrays with binary values.
        b: a batch of 3D arrays with binary values.
    kwargs:
        distance: the distance metric to use.
        spacing: the voxel spacing used.
    """
    assert a.shape == b.shape

    # Calculate Hausdorff distance for each item in batch.
    hd_dists = []
    for i in range(len(pred)):
        # Get item from batch.
        p, l = pred[i], label[i]

        # Get symmetric Hausdorff distance.
        hd_dist = hausdorff_distance(a, b, distance=distance, spacing=spacing)
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
    a_coords = np.argwhere(a != 0)
    b_coords = np.argwhere(b != 0)

    # 'np.argwhere' results in different shapes depending upon 'torch.Tensor' vs 'numpy.ndarray'.
    assert type(a) == type(b)
    if type(a) == torch.Tensor:
        a_coords = np.transpose(a_coords)
        b_coords = np.transpose(b_coords)

    # Shuffle coordinates, as this increases likelihood of early stopping.
    np.random.shuffle(a_coords)
    np.random.shuffle(b_coords)
    
    # Store the max distance (max) from a voxel in p to the closest point in l.
    max_min_dist = -np.inf
    
    for a_i in a_coords:
        # Convert to true spacing.
        a_true_i = a_i * np.array(spacing)
        
        # Store the minimum distance from a_true to any voxel in b.
        min_dist = np.inf
        
        for b_j in b_coords:
            # Convert to true spacing.
            b_true_j = b_j * np.array(spacing)
            
            # Find the distance between a_i and b_j.
            if distance == 'euclidean':
                dist = euclidean_distance(a_true_i, b_true_j)

            # Update the minimum distance if necessary.
            if dist < min_dist:
                min_dist = dist

                # Perform early stopping if we can't beat outer max.
                if min_dist < max_min_dist:
                    break

        # Update the maximum minimum distance if necessary.
        if min_dist > max_min_dist:
            max_min_dist = min_dist
        
    return max_min_dist

def euclidean_distance(a, b):
    """
    returns: the Euclidean distance between 3-dimensional points a and b.
    args:
        a: the first 3D point.
        b: the second 3D point.
    """
    assert len(a) == len(b)

    # Loop over elements.
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2

    return np.sqrt(sum)
