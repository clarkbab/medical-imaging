import subprocess

dry_run = False

regions = list(range(17))
models = [0, 1, 2, 5, 8]
models = [0]
random_seed = 42
target_region_idxs = [0, 1]
script = 'scripts/gradcam/nifti/spartan/array/heatmaps_pmcc_2_regions_numbers.slurm'

for model in models:
    for target_region_idx in target_region_idxs:
        # Create slurm command.
        export = f"ALL,RANDOM_SEED={random_seed},TARGET_REGION={target_region_idx}"
        command = f'sbatch --array={model} --export={export} {script}' 
        print(command)
        if not dry_run:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
