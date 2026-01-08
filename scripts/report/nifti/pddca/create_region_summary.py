from mymi.reports.nifti import create_region_summary
from mymi.utils import parse_arg

dataset = 'PDDCA'
region = parse_arg('region', str)
region = [
    'ts_OpticNrv_L',
    'ts_OpticNrv_R',
    'ts_Parotid_L',
    'ts_Parotid_R',
]
create_region_summary(dataset, region=region)
