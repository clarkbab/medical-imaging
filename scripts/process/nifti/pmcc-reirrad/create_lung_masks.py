from mymi.processing.nifti import combine_labels

dataset = 'PMCC-REIRRAD'
pat_ids = ['pat_4', 'pat_9']
study_ids = 'study_0'
regions = ['Lung_L', 'Lung_R']
output_region = 'Lungs'
combine_labels(dataset, pat_ids, study_ids, regions, output_region)