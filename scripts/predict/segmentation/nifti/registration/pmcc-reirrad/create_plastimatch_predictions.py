from mymi.predictions.nifti import create_plastimatch_predictions
from mymi.utils import grid_arg

pat_ids = grid_arg('pat_ids', arg_type=str, default='all')

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    create_coefs=False,
    pat_ids='PMCC_ReIrrad_L11',
)
create_plastimatch_predictions(dataset, **kwargs)
