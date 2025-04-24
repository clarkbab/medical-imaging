from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Sequence, Tuple, Union

Axis = Literal[0, 1, 2]
AxisName = Literal['sagittal', 'coronal', 'axial']
Pixel = Tuple[int, int]               # Required by 'Box'.
Voxel = Tuple[int, int, int]          # Required by 'Box'.
Point2D = Tuple[float, float]         # Required by 'Box'.
Point3D = Tuple[float, float, float]  # Required by 'Box'.
Box2D = Tuple[Pixel, Pixel]
Box3D = Tuple[Voxel, Voxel]
BoxMM2D = Tuple[Point2D, Point2D]
BoxMM3D = Tuple[Point3D, Point3D]
Channel = int
Channels = Union[Channel, Sequence[Channel], Literal['all']]
Colour = Union[str, Tuple[float, float, float]]
Image = np.ndarray      # Required by 'CtImage'.
CtImage = Image
DoseImage = Image
Extrema = Literal[0, 1]
GroupID = Union[int, float, str]
GroupIDs = Union[GroupID, Sequence[GroupID]]
Images = Union[Image, Sequence[Image]]
ImageSize2D = Tuple[int, int]
ImageSize3D = Tuple[int, int, int]
ImageSizeMM2D = Tuple[float, float]
ImageSizeMM3D = Tuple[float, float, float]
ImageSpacing2D = Tuple[float, float]
ImageSpacing3D = Tuple[float, float, float]
LabelImage = Image
Landmark = int
Landmarks = Union[Landmark, Sequence[Landmark], Literal['all']]
LandmarkData = pd.DataFrame
ModelCheckpoint = Union[str, Literal['best', 'last']]
ModelName = Tuple[str, str]
MrImage = Image
PatientID = str
PatientIDs = Union[PatientID, Sequence[PatientID], Literal['all']]
Region = str
Regions = Union[Region, Sequence[Region], Literal['all']]
RegionLabel = LabelImage
RegionData = Dict[Region, RegionLabel]
SampleID = int
SampleIDs = Union[SampleID, Sequence[SampleID], Literal['all']]
SeriesID = str
Split = Literal['train', 'validation', 'test']
Splits = Union[Split, Sequence[Split], Literal['all']]
SpartanPartition = Literal['feit-gpu-a100', 'gpu-a100', 'gpu-a100-short', 'gpu-h100', 'sapphire']
SpartanPartitions = Union[SpartanPartition, Sequence[SpartanPartition]]
StudyID = str
StudyIDs = Union[StudyID, Sequence[StudyID], Literal['all']]
TrainingInterval = str
