import os

from mymi import config
from mymi.plotting.nifti import plot_patients

dataset = 'HANSEG'
kwargs = dict(
    idx=0.5,
    pat_ids='group:0:2',
    regions='all',
    savepath=os.path.join(config.directories.files, 'MULTIORG', 'hanseg-g0.pdf'),
    show_legend=False,
    show_progress=True,
    study='all',
    views='all',
    window_mask=(-1024, None),
)
plot_patients(dataset, **kwargs)

kwargs['pat_ids'] = 'group:1:2'
kwargs['savepath'] = os.path.join(config.directories.files, 'MULTIORG', 'hanseg-g1.pdf')
plot_patients(dataset, **kwargs)
