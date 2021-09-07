import collections
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from scipy.ndimage import center_of_mass
from typing import Any, Callable, List, Optional, OrderedDict, Sequence, Tuple, Union

from mymi import cache
from mymi import config
from mymi import logging
from mymi import types

from .ct_series import CTSeries
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter

class RTSTRUCTSeries:
    def __init__(
        self,
        dataset: str,
        pat_id: types.PatientID,
        id: str,
        ct_from: Optional[str] = None,
        region_map: Optional[RegionMap] = None):
        """
        args:
            dataset: the dataset name.
            pat_id: the patient ID.
            id: the RTSTRUCT series ID.
        kwargs:
            region_map: the RegionMap object.
        """
        self._dataset = dataset
        self._pat_id = pat_id
        self._id = id
        self._ct_from = ct_from
        self._region_map = region_map
        self._path = os.path.join(config.directories.datasets, 'raw', dataset, 'hierarchical', 'data', pat_id, 'rtstruct', id)

        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"RTSTRUCT series '{id}' not found for patient '{pat_id}', dataset '{dataset}'.")

        # Check that DICOM is present.
        rtstructs = os.listdir(self._path)
        if len(rtstructs) != 1:
            raise ValueError(f"Expected 1 RTSTRUCT, got '{len(rtstructs)}' for series '{id}' for patient '{pat_id}', dataset '{dataset}'.")

        # Get reference CT series.
        ds = self._dataset if ct_from is None else ct_from
        rtstruct = self.get_rtstruct()
        ct_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
        self._ref_ct = CTSeries(ds, pat_id, ct_id)

    @property
    def id(self) -> str:
        return self._id

    @property
    def ref_ct(self) -> str:
        return self._ref_ct

    @cache.method('_dataset', '_pat_id', '_id')
    def list_regions(
        self,
        use_mapping: bool = True) -> List[str]:
        rtstruct = self.get_rtstruct()
        names = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))
        names = list(filter(lambda n: RTSTRUCTConverter.has_roi_data(rtstruct, n), names))
        if use_mapping and self._region_map:
            names = [self._region_map.to_internal(n) for n in names]
        return names

    def get_rtstruct(self) -> dcm.dataset.FileDataset:
        """
        returns: an RTSTRUCT DICOM object.
        """
        # Load RTSTRUCT.
        rtstructs = os.listdir(self._path)
        rtstruct = dcm.read_file(os.path.join(self._path, rtstructs[0]))

        return rtstruct

    @cache.method('_dataset', '_pat_id', '_id')
    def list_regions(
        self,
        clear_cache: bool = False,
        use_mapping: bool = True) -> List[str]:
        """
        returns: the patient's region names.
        kwargs:
            clear_cache: force the cache to clear.
            use_mapping: use region map if present.
        """
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region names.
        names = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        names = list(filter(lambda n: RTSTRUCTConverter.has_roi_data(rtstruct, n), names))

        # Map to internal names.
        if use_mapping and self._region_map:
            names = [self._region_map.to_internal(n) for n in names]

        return names

    def has_region(
        self,
        region: str,
        use_mapping: bool = True) -> bool:
        """
        returns: if the patient has the region.
        args:
            region: the region name.
        kwargs:
            clear_cache: force the cache to clear.
            use_mapping: use region map if present.
        """
        return region in self.list_regions(use_mapping=use_mapping)

    @cache.method('_dataset', '_pat_id', '_id')
    def region_data(
        self,
        clear_cache: bool = False,
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> OrderedDict:
        """
        returns: an OrderedDict[str, np.ndarray] of region names and data.
        kwargs:
            clear_cache: force the cache to clear.
            regions: the desired regions.
            use_mapping: use region map if present.
        """
        self._assert_requested_regions(regions, use_mapping=use_mapping)

        # Get region names - include unmapped as we need these to load RTSTRUCT regions later.
        unmapped_names = self.list_regions(use_mapping=False)
        names = self.list_regions(use_mapping=use_mapping)
        names = list(zip(names, unmapped_names))

        # Filter on requested regions.
        def fn(pair):
            name, _ = pair
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return name == regions
            else:
                return name in regions
        names = list(filter(fn, names))

        # Get reference CTs.
        cts = self._ref_ct.get_cts()

        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Add ROI data.
        region_dict = {}
        for name in names:
            # Get binary mask.
            try:
                data = RTSTRUCTConverter.get_roi_data(rtstruct, name[1], cts)
            except ValueError as e:
                logging.error(f"Caught error extracting data for region '{name[1]}', dataset '{self._dataset}', patient '{self._id}'.")
                logging.error(f"Error message: {e}")
                continue

            region_dict[name[0]] = data

        # Create ordered dict.
        ordered_dict = collections.OrderedDict((n, region_dict[n]) for n in sorted(region_dict.keys())) 

        return ordered_dict

    @cache.method('_dataset', '_pat_id', '_id')
    def region_summary(
        self,
        clear_cache: bool = False,
        columns: Union[str, Sequence[str]] = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame region summary information.
        kwargs:
            clear_cache: clear the cache.
            columns: the columns to return.
            regions: the desired regions.
            use_mapping: use region map if present.
        """
        # Define table structure.
        cols = {
            'region': str,
            'centroid-mm-x': float,
            'centroid-mm-y': float,
            'centroid-mm-z': float,
            'centroid-voxels-x': int,
            'centroid-voxels-y': int,
            'centroid-voxels-z': int,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
            'width-voxels-x': int,
            'width-voxels-y': int,
            'width-voxels-z': int,
        }
        cols = dict(filter(self._filter_on_dict_keys(columns, whitelist='region'), cols.items()))
        df = pd.DataFrame(columns=cols.keys())

        # Get region dict.
        region_data = self.region_data(clear_cache=clear_cache, regions=regions, use_mapping=use_mapping)

        # Get voxel offset/spacing.
        offset = self._ref_ct.offset(clear_cache=clear_cache)
        spacing = self._ref_ct.spacing(clear_cache=clear_cache)

        # Add info for each region.
        for name, data in region_data.items():
            # Find centre-of-mass.
            centroid = np.round(center_of_mass(data)).astype(int)

            # Convert COM to millimetres.
            mm_centroid = (centroid * spacing) + offset

            # Find bounding box co-ordinates.
            non_zero = np.argwhere(data != 0)
            mins = non_zero.min(axis=0)
            maxs = non_zero.max(axis=0)
            voxel_widths = maxs - mins

            # Convert voxel widths to millimetres.
            mm_widths = voxel_widths * spacing

            data = {
                'region': name,
                'centroid-mm-x': mm_centroid[0],
                'centroid-mm-y': mm_centroid[1],
                'centroid-mm-z': mm_centroid[2],
                'centroid-voxels-x': centroid[0],
                'centroid-voxels-y': centroid[1],
                'centroid-voxels-z': centroid[2],
                'width-mm-x': mm_widths[0],
                'width-mm-y': mm_widths[1],
                'width-mm-z': mm_widths[2],
                'width-voxels-x': voxel_widths[0],
                'width-voxels-y': voxel_widths[1],
                'width-voxels-z': voxel_widths[2]
            }
            data = dict(filter(self._filter_on_dict_keys(columns, whitelist='region'), data.items()))
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by region.
        df = df.sort_values('region').reset_index(drop=True)

        return df

    def _filter_on_dict_keys(
        self,
        keys: Union[str, Sequence[str]] = 'all',
        whitelist: Union[str, List[str]] = None) -> Callable[[Tuple[str, Any]], bool]:
        """
        returns: a function that filters out unneeded keys.
        kwargs:
            keys: description of required keys.
            whitelist: keys that are never filtered.
        """
        def fn(item: Tuple[str, Any]) -> bool:
            key, _ = item
            # Allow based on whitelist.
            if whitelist is not None:
                if type(whitelist) == str:
                    if key == whitelist:
                        return True
                elif key in whitelist:
                    return True
            
            # Filter based on allowed keys.
            if ((isinstance(keys, str) and (keys == 'all' or key == keys)) or
                ((isinstance(keys, list) or isinstance(keys, np.ndarray) or isinstance(keys, tuple)) and key in keys)):
                return True
            else:
                return False
        return fn

    def _assert_requested_regions(
        self,
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> None:
        if type(regions) == str:
            if regions != 'all' and not self.has_region(regions, use_mapping=use_mapping):
                raise ValueError(f"Requested region '{regions}' not present for RTSTRUCT series '{self._id}', patient '{self._pat_id}', dataset '{self._dataset.description}'.")
        elif hasattr(regions, '__iter__'):
            for region in regions:
                if not self.has_region(region, use_mapping=use_mapping):
                    raise ValueError(f"Requested region '{region}' not present for RTSTRUCT series '{self._id}', patient '{self._pat_id}', dataset '{self._dataset.description}'.")
        else:
            raise ValueError(f"Requested regions '{regions}' isn't 'str' or 'iterable'.")
