from typing import Tuple, Union

Point2D = Tuple[int, int]
Point3D = Tuple[int, int, int]
Box2D = Tuple[Point2D, Point2D]
Box3D = Tuple[Point3D, Point3D]
Colour = Union[str, Tuple[float, float, float]]
PatientID = Union[int, str]
PatientView = str,
PhysPoint2D = Tuple[float, float],
PhysPoint3D = Tuple[float, float, float]
ProcessedFolder = str,
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
TrainInterval = Union[int, str]
