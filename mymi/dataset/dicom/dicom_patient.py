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
from .dicom_series import DICOMModality
from .dicom_study import DICOMStudy
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter
from .rtstruct_series import RTSTRUCTSeries

class DICOMPatient:
    def __init__(
        self,
        dataset: 'DICOMDataset',
        id: types.PatientID,
        ct_from: Optional['DICOMPatient'] = None,
        load_default_series: bool = True,
        region_map: Optional[RegionMap] = None,
        trimmed: bool = False):
        """
        args:
            dataset: the DICOMDataset the patient belongs to.
            id: the patient ID.
        """
        if trimmed:
            self._global_id = f"{dataset} - {id} (trimmed)"
        else:
            self._global_id = f"{dataset} - {id}"
        self._ct_from = ct_from
        self._dataset = dataset
        self._id = id
        self._trimmed = trimmed
        self._region_map = region_map
        if trimmed:
            self._path = os.path.join(dataset.path, 'hierarchy', 'trimmed', 'data', id)
        else:
            self._path = os.path.join(dataset.path, 'hierarchy', 'data', id)

        # Check that patient ID exists.
        if not os.path.isdir(self._path):
            raise ValueError(f"Patient '{self}' not found.")
        
        if load_default_series:
            self._load_default_series()

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return self._global_id

    def _require_ct(
        fn: Callable) -> Callable:
        """
        returns: a wrapped function that ensures CTs are present.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            # Pass query to alternate dataset if required.
            if self._ct_from is not None:
                alt_patient = DICOMPatient(self._ct_from, self._id)
                alt_fn = getattr(alt_patient, fn.__name__)
                fn_def = getattr(type(self), fn.__name__)
                if type(fn_def) == property:
                    return alt_fn
                else:
                    return alt_fn()

            # Check CT folder exists.
            cts_path = os.path.join(self._path, 'ct')
            if not os.path.exists(cts_path):
                raise ValueError(f"No CTs found for patient '{self}'.")

            # Check that there is at least one CT.
            ct_files = os.listdir(cts_path)
            if len(ct_files) == 0:
                raise ValueError(f"No CTs found for patient '{self}'.")
            
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

    def list_studies(self) -> List[str]:
        return list(sorted(os.listdir(os.path.join(self._path))))

    def get_study(
        self,
        id: str) -> DICOMStudy:
        return DICOMStudy(self, id, region_map=self._region_map)

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
    
    def _load_default_series(self) -> None:
        # Preference the first study - all studies without RTSTRUCTs have been trimmed.
        # TODO: Add configuration to determine which (multiple?) RTSTRUCTs to select.
        study = self.get_study(self.list_studies()[0])
        if len(study.list_series('rtstruct')) == 0:
            print(str(study))
        rt_series = study.get_series(study.list_series('rtstruct')[0], 'rtstruct')
        self._default_rt_series = rt_series

    # Proxy to default RTSTRUCT series.

    def ct_offset(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.offset(*args, **kwargs)

    def ct_size(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.size(*args, **kwargs)

    def ct_spacing(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.spacing(*args, **kwargs)

    def ct_orientation(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.orientation(*args, **kwargs)

    def ct_slice_summary(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.slice_summary(*args, **kwargs)

    def ct_summary(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.summary(*args, **kwargs)

    def ct_data(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.data(*args, **kwargs)

    def get_cts(self, *args, **kwargs):
        return self._default_rt_series.ref_ct.get_cts(*args, **kwargs)
 
    def get_rtstruct(self, *args, **kwargs):
        return self._default_rt_series.get_rtstruct(*args, **kwargs)

    def has_region(self, *args, **kwargs):
        return self._default_rt_series.has_region(*args, **kwargs)

    def list_regions(self, *args, **kwargs):
        return self._default_rt_series.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self._default_rt_series.region_data(*args, **kwargs)

    def region_summary(self, *args, **kwargs):
        return self._default_rt_series.region_summary(*args, **kwargs)
