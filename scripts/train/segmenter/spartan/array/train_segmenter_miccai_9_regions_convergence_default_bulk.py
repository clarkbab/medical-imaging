import subprocess

random_seeds = [42, 43, 44]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence_default.slurm'

for random_seed in random_seeds:
    run_name = f"seed-{random_seed}"
    export = f'ALL,RANDOM_SEED={random_seed},RUN_NAME={run_name}'
    command = f'sbatch --array=0 --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
