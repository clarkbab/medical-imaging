import subprocess

model_path = "/data/projects/punim1413/mymi/models/"
regions = ['BrachialPlexus_L', 'BrachialPlexus_R', 'Brain', 'BrainStem', 'Cochlea_L', 'Cochlea_R', 'Lens_L', 'Lens_R',
'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'OralCavity', 'Parotid_L', 'Parotid_R', 'SpinalCord', 'Submandibular_L',
'Submandibular_R']


for region in regions:
    command = f"unimelb-mf-download --mf.config /home/baclark/.mediaflux/mflux.cfg --out {model_path} /projects/proj-4000_punim1413-1128.4.825/localiser-{region}"
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
