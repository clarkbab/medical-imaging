from mymi.datasets.nifti import rename_patients

dataset = 'PMCC-REIRRAD-CP'
pat_map = {
    'pat_0': 'PMCC_ReIrrad_L01',
    'pat_1': 'PMCC_ReIrrad_L02',
    'pat_2': 'PMCC_ReIrrad_L03',
    'pat_3': 'PMCC_ReIrrad_L06',
    'pat_4': 'PMCC_ReIrrad_L07',
    'pat_5': 'PMCC_ReIrrad_L08',
    'pat_6': 'PMCC_ReIrrad_L09',
    'pat_7': 'PMCC_ReIrrad_L10',
    'pat_8': 'PMCC_ReIrrad_L11',
    'pat_9': 'PMCC_ReIrrad_L14',
}
inv_pat_map = {v: k for k, v in pat_map.items()}
def rename_fn(pat_id: str) -> str:
    return pat_map[pat_id]
kwargs = dict(
    dry_run=False,
    rename_evaluations=False,
    rename_folders=False,
    rename_indexes=False,
    rename_reports=False,
)
rename_patients(dataset, rename_fn, **kwargs)