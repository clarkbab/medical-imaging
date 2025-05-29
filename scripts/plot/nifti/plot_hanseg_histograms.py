import os

from mymi import config
from mymi.plotting.datasets.nifti import plot_patient_histograms

dataset = 'HANSEG'
kwargs = dict(
    pat_ids='group:0:2',
    savepath=os.path.join(config.directories.files, 'MULTIORG', 'hanseg-hists-g0.pdf'),
    show_progress=True,
)
plot_patient_histograms(dataset, **kwargs)

kwargs['pat_ids'] = 'group:1:2'
kwargs['savepath'] = os.path.join(config.directories.files, 'MULTIORG', 'hanseg-hists-g1.pdf')
plot_patient_histograms(dataset, **kwargs)
