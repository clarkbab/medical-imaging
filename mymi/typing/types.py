import numpy as np
import pandas as pd
import pydicom as dcm
import SimpleITK as sitk
import torch
from typing import *

# By default, image refers to a volume and slice to 2D data.
# Don't specify dimensions for 3D data, e.g. just use CtImage. If dealing with 2D using CtSlice.
# If both are possible, use Union[CtImage, CtSlice].
# Should the type be responsible for showing the expected dimensions? It's probably easier than writing docs.
# ImageArray/Tensor refers 3D object, SliceArray/Tensor refers to 2D object. 
# How do we show that batch/channel dimensions are supported? Perhaps do this in docs, as it will only be a 
# small number of methods that allow this for convenience (e.g. fov, resample, transform).
# Plurals at end (e.g. ImageArrays) denote a list. Plurals in the middle,
# (e.g. ImagesArray) denote an extra dimension in the array.
# 'array' denotes numpy type, 'tensor' denotes pytorch tensor.
# Can we just use "CtImageArray" for numpy arrays and "CtImageArrayTensor" where necessary.
# As most of our code operatures with numpy arrays.

Number = Union[int, float]
Axis = Literal[0, 1, 2]
AxisName = Literal['sagittal', 'coronal', 'axial']
Pixel = Tuple[int, int]
Pixels = np.ndarray
Voxel = Tuple[int, int, int]          # Required by 'Box'.
Voxels = np.ndarray
Point2D = Tuple[float, float]         # Required by 'Box'.
Point3D = Tuple[float, float, float]  # Required by 'Box'.
Point = Union[Point2D, Point3D]
Points2D = np.ndarray
Points3D = np.ndarray
Points = Union[Points2D, Points3D]
PointArray = np.ndarray     # D
PointTensor = torch.Tensor
PointsArray = np.ndarray    # NxD
PointsTensor = torch.Tensor
Box2D = Tuple[Pixel, Pixel]
Box3D = Tuple[Voxel, Voxel]
Box = Union[Box2D, Box3D]
BoxMM2D = Tuple[Point2D, Point2D]
BoxMM3D = Tuple[Point3D, Point3D]
BoxMM = Union[BoxMM2D, BoxMM3D]
Channel = int
Channels = Union[Channel, List[Channel], Literal['all']]
Colour = Union[str, Tuple[float, float, float]]
CtDicom = dcm.dataset.FileDataset
SliceArray = np.ndarray     # CT slices.
CtSlice = SliceArray
ImageArray = np.ndarray     # CT volumes.
CtImageArray = ImageArray
CtImageArrays = Union[CtImageArray, List[CtImageArray]]
DatasetID = str
Data = np.ndarray
DicomModality = Literal['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']
DicomModalities = Union[DicomModality, List[DicomModality], Literal['all']]
DicomSOPInstanceUID = str
DirPath = str
ImageArrays = Union[ImageArray, List[ImageArray]]
ImageTensor = torch.tensor
ImageTensors = Union[ImageTensor, List[ImageTensor]]
DoseSlice = SliceArray
DoseImageArray = ImageArray
DoseImageArrays = Union[DoseImageArray, List[DoseImageArray]]
Extrema = Literal[0, 1]
FilePath = str
FilePaths = Union[FilePath, List[FilePath]]
Fov2D = BoxMM2D
Fov3D = BoxMM3D
Fov = Union[Fov2D, Fov3D]
GlobalID = str
GroupID = Union[int, float, str]
GroupIDs = Union[GroupID, List[GroupID]]
LabelArray = ImageArray
LabelTensor = ImageTensor
LandmarkID = str
LandmarkIDs = Union[LandmarkID, List[LandmarkID], Literal['all']]
LandmarkSeries = pd.Series
LandmarksFrame = pd.DataFrame
LandmarksFrameVox = pd.DataFrame
ModelCheckpoint = Union[str, Literal['best', 'last']]
ModelID = str
ModelIDs = Union[ModelID, List[ModelID], Literal['all']]
ModelName = Tuple[str, str]
MrDicom = dcm.dataset.FileDataset
MrSlice = SliceArray
MrImageArray = ImageArray
MrImageArrays = Union[MrImageArray, List[MrImageArray]]
NiftiSeriesID = str
NiftiModality = Literal['ct', 'dose', 'mr', 'regions']
PatientID = str
PatientIDs = Union[PatientID, List[PatientID], Literal['all']]
Region = str
RegionID = str
Regions = Union[Region, List[Region], Literal['all']]
RegionIDs = Union[RegionID, List[RegionID], Literal['all']]
RegionArray = LabelArray
RegionTensor = LabelTensor
RegionArrays = Dict[Region, RegionArray]
RtDoseDicom = dcm.dataset.FileDataset
RtPlanDicom = dcm.dataset.FileDataset
RtStructDicom = dcm.dataset.FileDataset
SampleID = int
SampleIDs = Union[SampleID, List[SampleID], Literal['all']]
SeriesID = str
SeriesIDs = Union[SeriesID, List[SeriesID], Literal['all']]
Size2D = Tuple[int, int]
Size3D = Tuple[int, int, int]
Size = Union[Size2D, Size3D]
SizeArray = np.ndarray
SizeTensor = torch.Tensor
SizeMM2D = Tuple[float, float]
SizeMM3D = Tuple[float, float, float]
SizeMM = Union[SizeMM2D, SizeMM3D]
SizeMMTensor = torch.Tensor
# Spacing refers to a simple tuple.
# In some instances, we have to specify that the function should take a tensor
# or array. We don't want to have to keep shifting data on/off gpus by converting
# between tuples and tensors in every function.
Spacing2D = Tuple[float, float]
Spacing3D = Tuple[float, float, float]
Spacing = Union[Spacing2D, Spacing3D]
SpacingArray = np.ndarray   # D
SpacingTensor = torch.Tensor
Split = Literal['train', 'validation', 'test']
Splits = Union[Split, List[Split], Literal['all']]
SpartanPartition = Literal['feit-gpu-a100', 'gpu-a100', 'gpu-a100-short', 'gpu-h100', 'sapphire']
SpartanPartitions = Union[SpartanPartition, List[SpartanPartition]]
StudyID = str
StudyIDs = Union[StudyID, List[StudyID], Literal['all']]
ImageTensor = torch.Tensor
ImageTensors = Union[ImageTensor, List[ImageTensor]]
ImageTensor3D = ImageTensor
TrainingInterval = str
Transform = sitk.Transform
VectorImageArray = np.ndarray   # Deformation field.
VectorImageArrays = Union[VectorImageArray, List[VectorImageArray]]
