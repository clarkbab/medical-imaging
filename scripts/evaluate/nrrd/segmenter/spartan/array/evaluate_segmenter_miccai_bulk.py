import subprocess

regions = '0-3'
script = 'scripts/evaluate/nrrd/segmenter/spartan/array/evaluate_segmenter_miccai_1_region_precision.slurm'
precisions = [32, 'bf16']
random_seeds = [42, 43]

# Create slurm command.
for precision in precisions:
    for random_seed in random_seeds:
        export = f'ALL,PRECISION={precision},RANDOM_SEED={random_seed}'
        command = f'sbatch --array={regions} --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
