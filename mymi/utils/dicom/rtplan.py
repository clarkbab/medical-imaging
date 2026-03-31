import pydicom as dcm

from mymi.typing import *

def from_rtplan_dicom(
    rtplan: FilePath | RtPlanDicom,
    ) -> Dict[str, Any]:
    if isinstance(rtplan, str):
        rtplan = dcm.dcmread(rtplan, force=False)

    # Get info.
    info = {}
    info['isocentre'] = tuple([float(i) for i in rtplan.BeamSequence[0].ControlPointSequence[0].IsocenterPosition])
    # info['couch-shift'] = tuple([
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLateralPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopVerticalPosition),
    #     float(rtplan.BeamSequence[0].ControlPointSequence[0].TableTopLongitudinalPosition),
    # ])

    return info
