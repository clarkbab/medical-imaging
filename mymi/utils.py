import json
import hashlib
import numpy as np
from typing import Any

from mymi.loaders import Loader

def encode(o: Any) -> str:
    return hashlib.sha1(json.dumps(o).encode('utf-8')).hexdigest()

def get_manifest():
    datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
    region = 'BrainStem'
    num_folds = 5
    num_train = 5
    test_fold = 0
    _, _, test_loader = Loader.build_loaders(datasets, region, load_test_origin=False, num_folds=num_folds, num_train=num_train, test_fold=test_fold)
    samples = []
    for ds_b, samp_b in iter(test_loader):
        samples.append((ds_b[0], samp_b[0].item()))
    return samples

def get_batch_centroids(label_batch, plane):
    """
    returns: the centroid location of the label along the plane axis, for each
        image in the batch.
    args:
        label_batch: the batch of labels.
        plane: the plane along which to find the centroid.
    """
    assert plane in ('axial', 'coronal', 'sagittal')

    # Move data to CPU.
    label_batch = label_batch.cpu()

    # Determine axes to sum over.
    if plane == 'axial':
        axes = (0, 1)
    elif plane == 'coronal':
        axes = (0, 2)
    elif plane == 'sagittal':
        axes = (1, 2)

    centroids = np.array([], dtype=np.int)

    # Loop through batch and get centroid for each label.
    for label_i in label_batch:
        # Get weighting along 'plane' axis.
        weights = label_i.sum(axes)

        # Get average weighted sum.
        indices = np.arange(len(weights))
        avg_weighted_sum = (weights * indices).sum() /  weights.sum()

        # Get centroid index.
        centroid = np.round(avg_weighted_sum).long()
        centroids = np.append(centroids, centroid)

    return centroids
