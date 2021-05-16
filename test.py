from mymi import dataset
from mymi import cache

dataset.select('PM-limbus', ct_from='PM')

pats = dataset.list_patients()

print(dataset.info(clear_cache=True))
