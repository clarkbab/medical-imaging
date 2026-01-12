import matplotlib as mpl
from typing import *

from mymi.datasets.nifti import NiftiDataset, NiftiImageSeries, NiftiLandmarksSeries, NiftiModality, NiftiPatient, NiftiRegionsSeries, NiftiStudy
from mymi.predictions.nifti import load_registration, load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from .data import plot_histogram as ph
from .patients import plot_patient_histogram as pph, plot_patient as pp
from .registration import plot_registration as pr
from .series import plot_series as ps

def plot_dataset_histogram(
    dataset: NiftiDataset,
    pat: PatientIDs = 'all',
    # TODO: Allow studies/series to be specified per patient?
    **kwargs) -> None:
    pat_ids = dataset.list_patients(pat=pat)
    n_rows = len(pat_ids)

    # Get the number of columns.
    n_cols = 0
    for p in pat_ids:
        pat = dataset.patient(p)
        n_cols_tmp = plot_patient_histogram(pat, return_n_cols=True, **kwargs)
        if n_cols_tmp > n_cols:
            n_cols = n_cols_tmp

    _, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for p, axs in zip(pat_ids, axs):
        pat = dataset.patient(p)
        plot_patient_histogram(pat, ax=axs, **kwargs)

# New API - takes a NiftiImageSeries only.
# Optional landmarks/regions overlaid on image.
def plot_series(
    series: NiftiImageSeries,
    landmarks: Optional[NiftiLandmarksSeries] = None,
    regions: Optional[NiftiRegionsSeries] = None,
) -> None:
    pass

def plot_series_histogram(
    series: NiftiImageSeries,
    **kwargs) -> None:
    ph(series.data, **kwargs)

# Takes a study and plots all CT series histograms.
def plot_study_histogram(
    study: NiftiStudy,
    modality: NiftiModality = 'ct',
    series: SeriesIDs = 'all',
    **kwargs) -> None:
    # Load image series IDs.
    series_ids = study.list_series(modality, series)
    n_series = len(series_ids)
    _, axs = plt.subplots(1, n_series, figsize=(4 * n_series, 4), squeeze=False)

    # Plot series.
    for s, ax in zip(series_ids, axs):
        series = study.series(s, modality)
        plot_series_histogram(series, ax=ax)

@delegates(pp)
def plot_patient(dataset, *args, **kwargs) -> None:
    set = NiftiDataset(dataset)
    fns = {
        'ct_series': lambda study, data_id: study.series(data_id, 'ct'),
        'default_dose': lambda study: study.default_dose,
        'default_landmarks': lambda study: study.default_landmarks,
        'default_regions': lambda study: study.default_regions,
        'dose_series': lambda study, series: study.series(series, 'dose'),
        'has_dose': lambda study: study.has_dose,
        'landmarks_data': lambda series, landmark_ids: series.data(landmark=landmark_ids),
        'landmark_series': lambda study, series: study.series(series, 'landmarks'),
        'regions_data': lambda series, region_ids: series.data(region=region_ids),
        'region_series': lambda study, series: study.series(series, 'regions'),
        'study_datetime': lambda study: None,  # Not currently available, might be able to store in nifti metadata: https://pycad.medium.com/store-the-metadata-in-nifti-and-nrrd-8aa4c6d942b5
    }
    pp(set, fns, *args, **kwargs)

def plot_patient_histogram(
    patient: NiftiPatient,
    ax: Optional[mpl.axes.Axes] = None,     # A row of patient axes.
    modality: NiftiModality = 'ct',
    return_n_cols: bool = False,
    series: SeriesIDs = 'all',
    study: StudyIDs = 'all',
    **kwargs) -> Optional[int]:
    # Determine study/series IDs.
    # TODO: Break this code out anywhere we need to determine study/series IDs.
    study_ids = patient.list_studies(study)
    if any(isinstance(s, list) for s in series):  # Series are specific to a study.
        assert len(series) == len(study_ids), f"Expected len(series) to match number of studies."
        # Expand study_ids to match number of passed series.
        new_study_ids = []
        for s, sr in zip(study_ids, series):
            new_study_ids += [s] * len(sr)
        study_ids = new_study_ids
        series_ids = series
    else:   # Series are shared across all studies.
        series_ids = arg_to_list(series, SeriesID, broadcast=len(study_ids))
        new_study_ids = []
        for s in study_ids:
            new_study_ids += [s] * len(series_ids)
        study_ids = new_study_ids

    n_series = len(series_ids)
    if return_n_cols:
        return n_series
    if ax is None:
        _, axs = plt.subplots(1, n_series, figsize=(4 * n_series, 4), squeeze=False)
    else:
        axs = ax

    # Plot studies/series.
    for s, sr, ax in zip(study_ids, series_ids, axs):
        # TODO: Could we use 'plot_study_histogram' here?
        series = patient.study(s).series(sr, modality)
        plot_series_histogram(series, ax=ax)

@delegates(pr)
def plot_registration(*args, **kwargs) -> None:
    fns = {
        'load_registration': load_registration,
    }
    pr(NiftiDataset, fns, *args, **kwargs)
