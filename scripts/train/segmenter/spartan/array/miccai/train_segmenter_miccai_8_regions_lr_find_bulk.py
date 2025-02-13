import subprocess

dry_run = False

regions = '0-1'
seeds = [42, 43, 44, 45, 46]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_8_regions_lr_find.slurm'

for seed in seeds:
    export = f'ALL,RANDOM_SEED={seed}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    if not dry_run:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
