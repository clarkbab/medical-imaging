from mymi.processing.datasets.nifti import create_vxmpp_preprocessed_dataset

dataset = 'DIRLAB-LUNG-COPD'
new_dataset = 'DIRLAB-LUNG-COPD-VXMPP'
kwargs = dict(
    lung_region='Lung',
)
create_vxmpp_preprocessed_dataset(dataset, new_dataset, **kwargs)
