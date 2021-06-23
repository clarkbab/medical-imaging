import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)
from mymi.dataset.dicom import ROIData, RTStructConverter
from mymi import types

def _create_sphere(
    size: types.ImageSize3D,
    centre: types.Point3D,
    radius: int) -> np.ndarray:
    """
    returns: a binary 3D numpy array containing a sphere.
    """
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(centre, size)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(size, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0

def test_bidirectional_conversion():
    # Create sphere.
    size = (10, 10, 10)
    centre = (5, 5, 5)
    radius = 3
    before = _create_sphere(size, centre, radius)

    # Perform bidirectional conversion.
    cts = []
    rtstruct = RTStructConverter.create_rtstruct(cts)
    roi_data = ROIData(
        data=before,
        frame_of_reference_uid='UID',
        name='Sphere'
    )
    RTStructConverter.add_roi(rtstruct, roi_data, cts)
    after = RTStructConverter.get_roi_data(rtstruct, 'Sphere', cts)

    # Assert that conversion doesn't alter the segmentation.
    np.testing.assert_array_equal(before, after)

test_bidirectional_conversion()
