from mymi.processing.dicom import convert_velocity_registrations_to_nifti

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    # pat=['PMCC_ReIrrad_L01', 'PMCC_ReIrrad_L02'],
    pat='PMCC_ReIrrad_L14',
    # method=['dmp', 'edmp', 'sg_c', 'sg_lm'],
    # method='dmp',
    region=None,
)
convert_velocity_registrations_to_nifti(dataset, **kwargs)
