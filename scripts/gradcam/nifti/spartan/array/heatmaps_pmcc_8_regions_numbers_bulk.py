import subprocess

dry_run = False

model_idxs = list(range(5))
model_idxs = [2]
target_regionxs = list(range(8))
target_regionxs = [0]
script = 'scripts/gradcam/nifti/spartan/array/heatmaps_pmcc_8_regions_numbers.slurm'

# Bash doesn't support nested arrays so we have to do this here.
TARGET_REGIONS = [
    ['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R','Cochlea_L','Cochlea_R','Lens_L','Lens_R'],    # 0
    ['Brain','OralCavity','BrainStem','Mandible','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R'],          # 1
    ['SpinalCord']                                                                                                      # 2
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
