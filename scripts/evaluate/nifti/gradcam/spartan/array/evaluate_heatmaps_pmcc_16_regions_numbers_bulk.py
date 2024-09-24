import subprocess

dry_run = False

model_idxs = list(range(2))
model_idxs = [0]
target_region_idxs = list(range(16))
target_region_idxs = [0]
script = 'scripts/evaluate/nifti/gradcam/spartan/array/evaluate_heatmaps_pmcc_16_regions_numbers.slurm'

# Bash doesn't support nested arrays so we have to do this here.
TARGET_REGIONS = [
    ['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Brain','OralCavity','BrainStem','Mandible','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R'],
    ['SpinalCord']
]

for model_idx in model_idxs:
    for target_region_idx in target_region_idxs:
        # Create slurm command.
        target_region = TARGET_REGIONS[model_idx][target_region_idx]
        export = f"ALL,TARGET_REGION={target_region}"
        command = f'sbatch --array={model_idx} --export={export} {script}' 
        print(command)
        if not dry_run:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
