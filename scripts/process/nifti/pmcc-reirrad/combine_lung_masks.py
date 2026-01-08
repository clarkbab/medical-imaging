from mymi.processing.nifti import combine_labels

dataset = 'PMCC-REIRRAD'
# pat_ids = ['PMCC_ReIrrad_L03', 'PMCC_ReIrrad_L07', 'PMCC_ReIrrad_L14']
# studys = 'study_0'
pat_ids = ['PMCC_ReIrrad_L21', 'PMCC_ReIrrad_L22', 'PMCC_ReIrrad_L23', 'PMCC_ReIrrad_L24', 'PMCC_ReIrrad_L25', 'PMCC_ReIrrad_L26', 'PMCC_ReIrrad_L27', 'PMCC_ReIrrad_L28', 'PMCC_ReIrrad_L29', 'PMCC_ReIrrad_L30']
studys = ['study_0', 'study_1']
regions = ['Lung_L', 'Lung_R']
output_region = 'Lungs'
dry_run = False
combine_labels(dataset, pat_ids, studys, regions, output_region, dry_run=dry_run)
