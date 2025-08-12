from mymi.processing.dicom import convert_to_nifti

dataset = 'PMCC-REIRRAD'
kwargs = dict(
    pat_ids='PMCC_ReIrrad_L01',
    recreate_dose=True,
)
convert_to_nifti(dataset, **kwargs)
