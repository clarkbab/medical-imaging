from mymi.reporting.nifti import create_region_summary
from mymi.utils import grid_arg

dataset = 'PDDCA'
region_id = grid_arg('region', str)
create_region_summary(dataset, region_ids=region_id)
