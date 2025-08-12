# cspm_grid evaluate_models.py --params 'model' --values "['corrfield','deeds','identity','plastimatch','unigradicon','unigradicon-io','velocity-dmp','velocity-edmp']"
from mymi.evaluations.nifti import create_registration_evaluations
from mymi.utils import grid_arg

model = grid_arg('model', str)

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    # pat_ids='idx:0',
    # region_ids=None,
    splits='test',
)

create_registration_evaluations(dataset, model, **kwargs)
