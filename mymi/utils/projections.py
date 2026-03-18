from augmed.typing import *
import numpy as np
import pydicom as dcm
from typing import Any, Dict, List, Tuple

from mymi.typing import DirPath, FilePath, SliceBatch, Volume

def load_dicom_projections(
    filepath: FilePath,
    ) -> Tuple[SliceBatch, Dict[str, Any]]:
    ds = dcm.dcmread(filepath)
    assert ds.PatientPosition == 'HFS'
    assert ds.RTImageOrientation == [1, 0, 0, 0, -1, 0]

    # Add info.
    info = {}
    info['sid'] = float(ds.RadiationMachineSAD)     
    info['sdd'] = float(ds.RTImageSID)
    info['det-spacing'] = tuple(float(f) for f in ds.ImagePlanePixelSpacing)
    info['det-offset'] = tuple(float(o) for o in ds.XRayImageReceptorTranslation[:2])

    kv_source_angles = []
    for i, f in enumerate(ds.ExposureSequence):
        frame = int(f.ReferencedFrameNumber)
        assert frame == i + 1
        angle = float(getattr(f, "GantryAngle", np.nan))
        kv_source_angles.append(angle)
    info['kv-source-angles'] = kv_source_angles

    # Load pixel data.
    data = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    data = data * slope + intercept

    # Transpose x/y axes.
    data = np.moveaxis(data, 1, 2)
    info['det-size'] = data.shape[1:]

    return data, info
