from mymi.processing.nifti import convert_registrations_from_lung_preprocessed

dataset = 'PMCC-REIRRAD-CP'
dest_dataset = 'PMCC-REIRRAD'
models=[
    # 'corrfield',
    # 'deeds',
    # 'identity',
    # 'plastimatch',
    # 'unigradicon',
    # 'unigradicon-io',
    'velocity-dmp',
    'velocity-edmp',
]
kwargs = dict(
    dry_run=False,
    pat_ids=['PMCC_ReIrrad_L08', 'PMCC_ReIrrad_L14'],
    # region_ids=None,
    # warp_ct=False,
)
convert_registrations_from_lung_preprocessed(dataset, dest_dataset, models, **kwargs)
