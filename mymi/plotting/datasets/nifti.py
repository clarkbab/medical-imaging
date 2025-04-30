from typing import *

from mymi.datasets.nifti import NiftiDataset
from mymi.predictions.datasets.nifti import load_registration as lr, load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from ..plotting import plot_dataset_histogram as pdh, plot_patient_histograms as pph, plot_patients as pp, plot_registrations as pr, plot_segmenter_predictions as plot_segmenter_predictions_base

def plot_dataset_histogram(*args, **kwargs) -> None:
    pdh(NiftiDataset, *args, **kwargs)

def plot_patients(*args, **kwargs) -> None:
    pp(NiftiDataset, *args, **kwargs)

def plot_patient_histograms(*args, **kwargs) -> None:
    pph(NiftiDataset, *args, **kwargs)

def plot_registrations(*args, **kwargs) -> None:
    pr(NiftiDataset, lr, *args, **kwargs)

def plot_segmenter_predictions(
    dataset: str,
    pat_id: str,
    model: str,
    centre: Optional[str] = None,
    crop: Optional[str] = None,
    regions: Regions = 'all',
    regions_model: Regions = 'all',
    study_id: str = 'study_0',
    **kwargs) -> None:
    
    # Load data.
    set = NiftiDataset(dataset)
    study = set.patient(pat_id).study(study_id)
    ct_data = study.ct_data
    spacing = study.ct_spacing
    region_data = study.region_data(regions=regions)

    # Load predictions.
    pred_data = load_segmenter_predictions(dataset, pat_id, model, regions=regions_model, study_id=study_id)

    # Only handle centre of ground truth - not pred.
    if isinstance(centre, str):
        assert isinstance(centre, str)
        if region_data is None or centre not in region_data:
            centre = study.region_data(regions=centre)[centre]

    if isinstance(crop, str):
        assert isinstance(crop, str)
        if region_data is None or crop not in region_data:
            crop = study.region_data(regions=crop)[crop]
    
    # Plot.
    okwargs = dict(
        centre=centre,
        crop=crop,
        ct_data=ct_data,
        region_data=region_data,
        **kwargs
    )
    plot_segmenter_predictions_base(pat_id, spacing, pred_data, **okwargs)
