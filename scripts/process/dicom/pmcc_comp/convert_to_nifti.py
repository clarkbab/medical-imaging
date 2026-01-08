from mymi.processing.dicom import convert_to_nifti

dataset = 'PMCC-COMP'
kwargs = dict(
    regions='rl:pmcc-comp'
)
convert_to_nifti(dataset, **kwargs)
