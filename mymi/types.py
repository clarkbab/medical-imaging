from typing import Literal, Tuple

Box2D = Tuple[Tuple[int, int], Tuple[int, int]]
Box3D = Tuple[Tuple[int, int, int], Tuple[int, int, int]]
Size3D = Tuple[int, int, int]
Spacing3D = Tuple[float, float, float]
PatientView = Literal['axial', 'coronal', 'sagittal']
