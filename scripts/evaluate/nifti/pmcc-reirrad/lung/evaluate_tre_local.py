# cspm_grid evaluate_models.py --params 'model' --values "['corrfield','deeds','identity','plastimatch','unigradicon','unigradicon-io','velocity-dmp','velocity-edmp']"
from mymi.evaluations.nifti import create_registration_evaluation
from mymi.utils import parse_arg

model = parse_arg('model', str)
model = ['velocity-dmp', 'velocity-edmp', 'velocity-rir', 'velocity-sg_c', 'velocity-sg_lm']
# model = 'velocity-rir'

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    group='lung',
    # pat=['PMCC_ReIrrad_L01', 'PMCC_ReIrrad_L02'],
    pat='i:2',
    region=None,
)

create_registration_evaluation(dataset, model, **kwargs)
