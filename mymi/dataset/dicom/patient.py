import numpy as np
from numpy import ndarray
import os
from pandas import DataFrame
import pydicom as dicom
from pydicom.dataset import FileDataset
from scipy.ndimage import center_of_mass
from skimage.draw import polygon
from typing import *

from mymi import cache
from mymi.cache import cached_method
from mymi import config

class Patient:
    def __init__(
        self,
        dataset: str,
        id: str):
        """
        args:
            dataset: the dataset the patient belongs to, e.g. 'HEAD-NECK-RADIOMICS-HN1'.
            id: the patient ID.
        """
        self._dataset = dataset
        self._id = id
        self._path = os.path.join(config.directories.datasets, dataset, 'hierarchical', id)

    def ct_dicoms(self) -> Sequence[FileDataset]:
        """
        returns: a list of FileDataset objects holding CT info.
        """
        # Load all CT dicoms.
        cts_path = os.path.join(self._path, 'ct')
        ct_paths = [os.path.join(cts_path, f) for f in os.listdir(cts_path)]
        cts = [dicom.read_file(f) for f in ct_paths]

        return cts

    def rtstruct_dicom(self) -> FileDataset:
        """
        returns: a FileDataset object holding RTSTRUCT info.
        """
        # Load RTSTRUCT dicom.
        rtstructs_path = os.path.join(self._path, 'rtstruct')
        rtstruct_paths = [os.path.join(rtstructs_path, f) for f in os.listdir(rtstructs_path)]
        if len(rtstruct_paths) != 1:
            raise ValueError(f"Expected 1 RTSTRUCT dicom for patient '{self._id}', got {len(rtstruct_paths)}.")
        rtstruct = dicom.read_file(rtstruct_paths[0])

        return rtstruct

    @cached_method('_dataset', '_id')
    def ct_summary(self) -> DataFrame:
        """
        returns: a table summarising CT info.
        """
        # Define dataframe structure.
        cols = {
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'size-x': int,
            'size-y': int,
            'size-z': int,
            'spacing-x': float,
            'spacing-y': float,
        }
        df = DataFrame(columns=cols.keys())

        # Load CT dicoms.
        cts = self.ct_dicoms()

        # Add summary.
        data = {}
        z_offsets = []
        for ct in cts:
            # Add HU stats.
            hus = ct.pixel_array * ct.RescaleSlope + ct.RescaleIntercept
            hu_min = hus.min()
            hu_max = hus.max()
            if 'hu-min' not in data or hu_min < data['hu-min']:
                data['hu-min'] = hu_min
            if 'hu-max' not in data or hu_max > data['hu-max']:
                data['hu-max'] = hu_max

            # Add offsets.
            x_offset = ct.ImagePositionPatient[0]
            y_offset = ct.ImagePositionPatient[1]
            z_offset = ct.ImagePositionPatient[2]
            z_offsets.append(z_offset)
            if 'offset-x' not in data:
                data['offset-x'] = x_offset
            elif x_offset != data['offset-x']:
                raise ValueError(f"Inconsistent 'offset-x' for patient '{self._id}' CT scans.")
            if 'offset-y' not in data:
                data['offset-y'] = y_offset
            elif y_offset != data['offset-y']:
                raise ValueError(f"Inconsistent 'offset-y' for patient '{self._id}' CT scans.")
            if 'offset-z' not in data or z_offset < data['offset-z']:
                data['offset-z'] = z_offset

            # Add sizes.
            x_size = ct.pixel_array.shape[1]
            y_size = ct.pixel_array.shape[0]
            if 'size-x' not in data:
                data['size-x'] = x_size
            elif x_size != data['size-x']:
                raise ValueError(f"Inconsistent 'size-x' for patient '{self._id}' CT scans.")
            if 'size-y' not in data:
                data['size-y'] = y_size
            elif y_size != data['size-y']:
                raise ValueError(f"Inconsistent 'size-y' for patient '{self._id}' CT scans.")

            # Add x/y-spacings.
            x_spacing = ct.PixelSpacing[0]
            y_spacing = ct.PixelSpacing[1]
            if 'spacing-x' not in data:
                data['spacing-x'] = x_spacing
            elif x_spacing != data['spacing-x']:
                raise ValueError(f"Inconsistent 'spacing-x' for patient '{self._id}' CT scans.")
            if 'spacing-y' not in data:
                data['spacing-y'] = y_spacing
            elif y_spacing != data['spacing-y']:
                raise ValueError(f"Inconsistent 'spacing-y' for patient '{self._id}' CT scans.")

        # Add z-spacing.
        z_spacing = np.min([round(s, 2) for s in np.diff(sorted(z_offsets))])
        data['spacing-z'] = z_spacing

        # Add fields-of-view.
        x_fov = data['size-x'] * data['spacing-x']
        y_fov = data['size-y'] * data['spacing-y']
        z_fov = np.max(z_offsets) - np.min(z_offsets)
        data['fov-x'] = x_fov
        data['fov-y'] = y_fov
        data['fov-z'] = z_fov

        # Add z-size.
        z_size = int(round(z_fov / z_spacing, 0) + 1)
        data['size-z'] = z_size

        # Add row.
        df = df.append(data, ignore_index=True)
        df = df.reindex(sorted(df.columns), axis=1)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    @cached_method('_dataset', '_id')
    def label_summary(
        self,
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all') -> DataFrame:
        """
        returns: a DataFrame label summary information.
        kwargs:
            clear_cache: clear the cache.
            labels: the desired labels.
        """
        # Define table structure.
        cols = {
            'label': str,
            'com-x': int,
            'com-y': int,
            'com-z': int,
            'width-x': float,
            'width-y': float,
            'width-z': float,
        }
        df = DataFrame(columns=cols.keys())

        # Get label (name, data) pairs.
        label_data = self.label_data(clear_cache=clear_cache, labels=labels)

        # Get voxel spacings.
        summary = self.ct_summary(clear_cache=clear_cache).iloc[0].to_dict()
        spacing = (summary['spacing-x'], summary['spacing-y'], summary['spacing-z'])

        # Add info for each label.
        for name, data in label_data:
            # Find centre-of-mass.
            coms = np.round(center_of_mass(data)).astype(int)

            # Find bounding box co-ordinates.
            non_zero = np.argwhere(data != 0)
            mins = non_zero.min(axis=0)
            maxs = non_zero.max(axis=0)
            voxel_widths = maxs - mins

            # Convert voxel widths to millimetres.
            widths = voxel_widths * spacing

            data = {
                'label': name,
                'com-x': coms[0],
                'com-y': coms[1],
                'com-z': coms[2],
                'width-x': widths[0],
                'width-y': widths[1],
                'width-z': widths[2]
            }
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by label.
        df = df.sort_values('label').reset_index(drop=True)

        return df

    @cached_method('_dataset', '_id')
    def ct_data(self) -> ndarray:
        """
        returns: a 3D numpy ndarray of CT data in HU.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Load patient CT dicoms.
        cts = self.ct_dicoms()
        summary = self.ct_summary().iloc[0].to_dict()
        
        # Create CT data array.
        shape = (int(summary['size-x']), int(summary['size-y']), int(summary['size-z']))
        data = np.zeros(shape=shape)
        for ct in cts:
            # Convert to HU. Transpose to (x, y) coordinates, 'pixel_array' returns
            # row-fist image data.
            ct_data = np.transpose(ct.pixel_array)
            ct_data = ct.RescaleSlope * ct_data + ct.RescaleIntercept

            # Get z index.
            z_offset =  ct.ImagePositionPatient[2] - summary['offset-z']
            z_idx = int(round(z_offset / summary['spacing-z']))

            # Add data.
            data[:, :, z_idx] = ct_data

        return data

    @cached_method('_dataset', '_id')
    def label_data(
        self,
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all') -> Sequence[Tuple[str, ndarray]]:
        """
        returns: a list of (name, data) pairs, one for each label.
        kwargs:
            clear_cache: force the cache to clear.
            labels: the desired labels.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.rtstruct_dicom()

        # Get label shape.
        summary = self.ct_summary(clear_cache=clear_cache).iloc[0].to_dict()
        shape = (int(summary['size-x']), int(summary['size-y']), int(summary['size-z']))

        # Convert label data from vertices to voxels.
        contours = rtstruct.ROIContourSequence
        infos = rtstruct.StructureSetROISequence
        label_pairs = []
        for contour, info in zip(contours, infos):
            # Get contour name.
            name = info.ROIName

            # Skip if label not needed.
            if not (labels == 'all' or
                ((type(labels) == tuple or type(labels) == list) and name in labels) or
                (type(labels) == str and name == labels)):
                continue

            # Create label placeholder.
            data = np.zeros(shape=shape, dtype=bool)

            # Convert vertices into voxel data. 
            all_vertices = [c.ContourData for c in contour.ContourSequence]
            for vertices in all_vertices:
                # Coords are stored in flat array.
                vertices = np.array(vertices).reshape(-1, 3)

                # Convert from physical coordinates to voxel coordinates.
                x_indices = (vertices[:, 0] - summary['offset-x']) / summary['spacing-x']
                y_indices = (vertices[:, 1] - summary['offset-y']) / summary['spacing-y']

                # Get all voxels on the boundary and interior described by the vertices.
                x_indices, y_indices = polygon(x_indices, y_indices)

                # Get contour z pixel.
                z_offset = vertices[0, 2] - summary['offset-z']
                z_idx = int(z_offset / summary['spacing-z'])

                # Set labelled pixels in slice.
                data[x_indices, y_indices, z_idx] = 1

            label_pairs.append((name, data))

        # Sort by label name.
        label_pairs = sorted(label_pairs, key=lambda l: l[0])

        return label_pairs
