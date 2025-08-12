from mymi.datasets.dicom import rename_patients

dataset = 'PMCC-REIRRAD'
def rename_fn(old_pat_id: str) -> str:
    parts = old_pat_id.split('_')
    if parts[-1].startswith('L'):
        raise ValueError(f"Patient ID {old_pat_id} already has 'L' prefix.")
    parts[-1] = f'L{parts[-1]}'
    new_pat_id = '_'.join(parts)
    return new_pat_id
# pat_map = {'pat_0': 'PMCC_ReIrrad_L01',
#  'pat_1': 'PMCC_ReIrrad_L02',
#  'pat_2': 'PMCC_ReIrrad_L03',
#  'pat_3': 'PMCC_ReIrrad_L06',
#  'pat_4': 'PMCC_ReIrrad_L07',
#  'pat_5': 'PMCC_ReIrrad_L08',
#  'pat_6': 'PMCC_ReIrrad_L09',
#  'pat_7': 'PMCC_ReIrrad_L10',
#  'pat_8': 'PMCC_ReIrrad_L11',
#  'pat_9': 'PMCC_ReIrrad_L14'}
# def rename_fn(old_pat_id: str) -> str:
#     return pat_map.get(old_pat_id, old_pat_id)
kwargs = dict(
    dry_run=False,
    pat_regexp=r'PMCC_ReIrrad_[\d]+',
    # pat_regexp=r'pat_[\d]+',
)
rename_patients(dataset, rename_fn, **kwargs)
