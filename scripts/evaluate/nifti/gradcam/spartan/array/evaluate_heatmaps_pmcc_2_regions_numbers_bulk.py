import subprocess

dry_run = False

model_idx = list(range(9))
# model_idxs = [0]

target_regionxs = list(range(2))
# target_regionxs = [1]
script = 'scripts/evaluate/nifti/gradcam/spartan/array/evaluate_heatmaps_pmcc_2_regions_numbers.slurm'

# Bash doesn't support nested arrays so we have to do this here.
TARGET_REGIONS = [
    ['BrachialPlexus_L', 'BrachialPlexus_R'],     # 0
    ['Brain', 'OralCavity'],                      # 1
    ['BrainStem', 'Mandible'],                    # 2
    ['Cochlea_L', 'Cochlea_R'],                   # 3
    ['Lens_L', 'Lens_R'],                         # 4
    ['OpticNerve_L', 'OpticNerve_R'],             # 5
    ['Parotid_L', 'Parotid_R'],                   # 6
    ['SpinalCord', 'BrainStem'],                  # 7
    ['Submandibular_L', 'Submandibular_R']        # 8
]

for model_idx in model_idxs:
    for target_regionx in target_regionxs:
        # Create slurm command.
        target_region = TARGET_REGIONS[model_idx][target_regionx]
        export = f"ALL,TARGET_REGION={target_region}"
        command = f'sbatch --array={model_idx} --export={export} {script}' 
        print(command)
        if not dry_run:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
