from typing import Literal, Tuple, Union

Box2D = Tuple[Tuple[int, int], Tuple[int, int]]
Box3D = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
PatientID = Union[int, str]
PatientView = Literal['axial', 'coronal', 'sagittal']
Point2D = Tuple[float, float],
Point3D = Tuple[float, float, float]
ProcessedFolder = Literal['train', 'validate', 'test']
Size2D = Tuple[int, int]
Size3D = Tuple[int, int, int]
Spacing2D = Tuple[float, float]
Spacing3D = Tuple[float, float, float]
