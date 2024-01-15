import numpy as np

from mymi.geometry import get_extent
from mymi.transforms import resample

def interpolate_z(data: np.ndarray) -> np.ndarray:
    # Find "missing" slices.
    extent = get_extent(data)
    z_min = extent[0][2]
    z_max = extent[1][2]

    # Interpolate using ground truth data only, i.e. don't update the 
    # data as we go.
    new_data = data.copy()
    for z in range(z_min, z_max + 1):
        data_z = data[:, :, z]
        if data_z.sum() != 0:
            continue
            
        # Find closest non-empty slices.
        max_diff = 5
        data_below = None
        data_above = None
        for i in range(max_diff):
            if data_below is None and data[:, :, (z - i - 1)].sum() != 0:
                data_below = data[:, :, (z - i - 1)]
            if data_above is None and data[:, :, (z + i + 1)].sum() != 0:
                data_above = data[:, :, (z + i + 1)]
            if data_below is not None and data_above is not None:
                break
                
        if data_below is None or data_above is None:
            raise ValueError("")
            
        # Interpolate from surrounding slices.
        data_z_around = np.stack((data_below, data_above), axis=2)
        spacing = (1, 1, 1)
        output_spacing = (1, 1, 0.5)
        output_size = (*data.shape[0:2], 3)
        data_z_resampled = resample(data_z_around, output_size=output_size, output_spacing=output_spacing, spacing=spacing)
        
        # Replace slice.
        new_data[:, :, z] = data_z_resampled[:, :, 1]

    return new_data
