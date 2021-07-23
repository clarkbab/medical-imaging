import collections
import logging
from re import I
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import Any, Callable, List, Optional, OrderedDict, Sequence, Tuple, Union

from mymi import cache
from mymi import config
from mymi import types

from .ct_series import CTSeries
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter
from .rtstruct_series import RTSTRUCTSeries

class DicomPatient:
    def __init__(
        self,
        dataset: str,
        id: types.PatientID,
        ct_from: str = None,
        region_map: Optional[RegionMap] = None):
        """
        args:
            dataset: the dataset the patient belongs to, e.g. 'HEAD-NECK-RADIOMICS-HN1'.
            id: the patient ID.
        """
        self._ct_from = ct_from
        self._dataset = dataset
        self._id = id
        self._path = os.path.join(config.directories.datasets, dataset, 'hierarchical', 'data', id)
        self._region_map = region_map

        # Check number of RTSTRUCT series.
        rtstruct_series = self.list_rtstruct_series()
        if len(rtstruct_series) == 0:
            raise ValueError(f"Expected 1 RTSTRUCT, got '{len(rtstruct_series)}' for patient '{id}', dataset '{dataset}'.")
        
        # Set default RTSTRUCT series.
        self._default_rtstruct_series = RTSTRUCTSeries(dataset, id, rtstruct_series[0], ct_from=ct_from, region_map=region_map)

    def _require_ct(
        fn: Callable) -> Callable:
        """
        returns: a wrapped function that ensures CTs are present.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            # Pass query to alternate dataset if required.
            if self._ct_from:
                alt_patient = DicomPatient(self._ct_from, self._id)
                alt_fn = getattr(alt_patient, fn.__name__)
                fn_def = getattr(type(self), fn.__name__)
                if type(fn_def) == property:
                    return alt_fn
                else:
                    return alt_fn()

            # Check CT folder exists.
            cts_path = os.path.join(self._path, 'ct')
            if not os.path.exists(cts_path):
                raise ValueError(f"No CTs found for dataset '{self._dataset}', patient '{self._id}'.")

            # Check that there is at least one CT.
            ct_files = os.listdir(cts_path)
            if len(ct_files) == 0:
                raise ValueError(f"No CTs found for dataset '{self._dataset}', patient '{self._id}'.")
            
            return fn(self, *args, **kwargs)
        return wrapper

    def _use_internal_regions(
        fn: Callable) -> Callable:
        """
        returns: a wrapped function that renames DataFrame 'regions' to internal names.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            # Determine if internal region names are required.
            use_internal = kwargs.pop('internal_regions', False)

            # Call function.
            result = fn(self, *args, **kwargs)

            if use_internal:
                # Load region map.
                pass 
            else:
                return result

        return wrapper

    @property
    @_require_ct
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    @_require_ct
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate

    @property
    def ct_from(self) -> str:
        return self._ct_from

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def id(self) -> str:
        return self._id

    @property
    @_require_ct
    def name(self) -> str:
        return self.get_cts()[0].PatientName

    @property
    @_require_ct
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    @_require_ct
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    @_require_ct
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

    def list_ct_series(self) -> List[str]:
        """
        returns: CT series names for the patient.
        """
        # List the CT series.
        if self._ct_from is None:
            series = list(sorted(os.listdir(os.path.join(self._path, 'ct'))))
        else:
            series = DicomPatient(self._ct_from, self._id).list_ct_series()

        return series

    def ct_series(
        self,
        id: str) -> CTSeries:
        """
        returns: a CTSeries object.
        args:
            id: the CT series ID.
        """
        # Check that series ID exists.
        if not id in self.list_ct_series():
            raise ValueError(f"CT series '{id}' not found for patient '{self._id}', dataset '{self._dataset}'.")

        # Create CT series.
        ds = self._dataset if self._ct_from is None else self._ct_from
        series = CTSeries(ds, self._id, id)

        return series

    def list_rtstruct_series(self) -> List[str]:
        """
        returns: RTSTRUCT series names for the patient.
        raises:
            ValueError: when patient 'rtstruct' folder not found.
        """
        # Check that 'rtstruct' folder exists. This is required for a patient.
        rtstruct_path = os.path.join(self._path, 'rtstruct')
        if not os.path.exists(rtstruct_path):
            raise ValueError(f"No RTSTRUCTs found for dataset '{self._dataset}', patient '{self._id}'.")
    
        # Load RTSTRUCT series.
        series = list(sorted(os.listdir(os.path.join(self._path, 'rtstruct'))))
        return series

    def rtstruct_series(
        self,
        id: str) -> RTSTRUCTSeries:
        """
        returns: a RTSTRUCTSeries object.
        args:
            id: the RTSTRUCT series ID.
        """
        # Check that series ID exists.
        series_path = os.path.join(self._path, 'rtstruct', id)
        if not os.path.isdir(series_path):
            raise ValueError(f"RTSTRUCT series '{id}' not found for patient '{self._id}', dataset '{self._dataset}'.")

        # Create RTSTRUCT series.
        series = RTSTRUCTSeries(self._dataset, self._id, id, region_map=self._region_map)

        return series

    @_require_ct
    def info(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        """
        returns: a table of patient info.
        """
        # Define dataframe structure.
        cols = {
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

        # Add row.
        df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    # Proxy to default CTSeries.

    def get_cts(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.get_cts(*args, **kwargs)

    def ct_offset(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.offset(*args, **kwargs)

    def ct_size(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.size(*args, **kwargs)

    def ct_spacing(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.spacing(*args, **kwargs)

    def ct_slice_summary(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.slice_summary(*args, **kwargs)

    def ct_summary(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.summary(*args, **kwargs)

    def ct_data(self, *args, **kwargs):
        return self._default_rtstruct_series.ref_ct.data(*args, **kwargs)

    # Proxy to default RTSTRUCTSeries.

    def get_rtstruct(self, *args, **kwargs):
        return self._default_rtstruct_series.get_rtstruct(*args, **kwargs)

    def region_names(self, *args, **kwargs):
        return self._default_rtstruct_series.region_names(*args, **kwargs)

    def has_region(self, *args, **kwargs):
        return self._default_rtstruct_series.has_region(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self._default_rtstruct_series.region_data(*args, **kwargs)

    def region_summary(self, *args, **kwargs):
        return self._default_rtstruct_series.region_summary(*args, **kwargs)
