from mymi.processing.dataset.nifti import convert_to_training

dataset = 'PMCC-HN-REPLAN'
output_spacing = (1.171875, 1.171875, 2)
regions = ['Bone_Mandible', 'BrachialPlex_L', 'BrachialPlex_R', 'Brain', 'Brainstem', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R', 'Esophagus_S', 'Eye_L', 'Eye_R', 'GTVp', 'Glnd_Submand_L', 'Glnd_Submand_R', 'Glottis', 'Larynx', 'Lens_L', 'Lens_R', 'Musc_Constrict', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L', 'Parotid_R', 'SpinalCord']
use_registration = True

convert_to_training(dataset, output_spacing=output_spacing, region=regions, use_registration=use_registration)
