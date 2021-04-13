from scipy.spatial.distance import directed_hausdorff

def batch_hausdorff_distance(pred, label, spacing=(1.0, 1.0, 1.0)):
    """
    returns: the mean Hausdorff distance across the batch.
    args:
        pred: the batch of binary network predictions, e.g. shape (n, 512, 512, 212).
        label: the batch of labels, e.g. shape (n, 512, 512, 212)
        spacing: the voxel spacing used.
    """
    assert pred.shape == label.shape

    # Calculate Hausdorff distance for each item in batch.
    hd_dists = []
    for i in range(len(pred)):
        # Get item from batch.
        p, l = pred[i], label[i]

        # Get symmetric Hausdorff distance.
        hd_dist = max(directed_hausdorff_distance(p, l, spacing=spacing), directed_hausdorff_distance(l, p, spacing=spacing))
        hd_dists.append(hd_dist)

    return np.mean(hd_dists)

def directed_hausdorff_distance(volume_a, volume_b, spacing=(1.0, 1.0, 1.0)):
    """
    returns: the directed
    """
        # Store the max distance (max) from a voxel in p to the closest point in l.
        max_min_dist = None
        # Convert from binary values to (x, y, z) coordinates of non-zero voxels. Take care to convert
        # these coordinates to euclidean space using the 'spacing' parameter.
        # For each voxel in p (p_i).
            # Store the min distance (min_i) from p_i to a voxel in l.
            min_dist = None
            # For each voxel in l (l_j).
                # Find the euclidean distance between p_i and l_j.
                dist = ...

                # Update the minimum distance if necessary.
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            
            # Update the maximum minimum distance if necessary.
            if max_min_dist is None or min_dist > max_min_dist:
                max_min_dist = min_dist

            # Reset the minimum distance.
            min_dist = None
