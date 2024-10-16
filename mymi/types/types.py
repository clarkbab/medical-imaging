import pytorch_lightning as pl
from typing import List, Literal, Sequence, Tuple, Union

Axis = Literal[0, 1, 2]
AxisName = Literal['sagittal', 'coronal', 'axial']
Colour = Union[str, Tuple[float, float, float]]
Extrema = Literal[0, 1]
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSizeMM2D = Tuple[float, float]
ImageSizeMM3D = Tuple[float, float, float]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
ModelName = Tuple[str, str, str]
Model = pl.LightningModule
PatientID = str
PatientIDs = Union[PatientID, Sequence[PatientID]]
PatientView = Literal['axial', 'sagittal', 'coronal'],
PatientRegion = str
PatientRegions = Union[PatientRegion, List[PatientRegion]]
Point2D = Tuple[int, int]
Point3D = Tuple[int, int, int]
Box2D = Tuple[Point2D, Point2D]
Box3D = Tuple[Point3D, Point3D]
PointMM2D = Tuple[float, float]
PointMM3D = Tuple[float, float, float]
BoxMM2D = Tuple[PointMM2D, PointMM2D]
BoxMM3D = Tuple[PointMM3D, PointMM3D]
StudyID = str
TrainingPartition = Literal['train', 'validation', 'test']
TrainInterval = Union[int, str]