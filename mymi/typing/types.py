from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Sequence, Tuple, Union

class Axis(Enum):
    X = 0
    Y = 1
    Z = 2
AxisName = Literal['sagittal', 'coronal', 'axial']
Colour = Union[str, Tuple[float, float, float]]
Extrema = Literal[0, 1]
GroupID = Union[int, float, str]
GroupIDs = Union[GroupID, Sequence[GroupID]]
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSizeMM2D = Tuple[float, float]
ImageSizeMM3D = Tuple[float, float, float]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
Landmarks = pd.DataFrame
ModelName = Tuple[str, str, str]
PatientID = str
PatientIDs = Union[PatientID, Sequence[PatientID]]
PatientLandmark = str
PatientLandmarks = Union[PatientLandmark, Sequence[PatientLandmark], Literal['all']]
PatientRegion = str
PatientRegions = Union[PatientRegion, Sequence[PatientRegion], Literal['all']]
PatientView = Literal['axial', 'sagittal', 'coronal'],
Point2D = Tuple[int, int]
Point3D = Tuple[int, int, int]
PointMM2D = Tuple[float, float]
PointMM3D = Tuple[float, float, float]
Box2D = Tuple[Point2D, Point2D]
Box3D = Tuple[Point3D, Point3D]
BoxMM2D = Tuple[PointMM2D, PointMM2D]
BoxMM3D = Tuple[PointMM3D, PointMM3D]
SeriesID = str
StudyID = str
TrainingPartition = Literal['train', 'validation', 'test']
TrainInterval = Union[int, str]

Image = np.ndarray
CtImage = Image
LabelImage = Image
RegionImage = LabelImage
RegionImages = Dict[str, RegionImage]
