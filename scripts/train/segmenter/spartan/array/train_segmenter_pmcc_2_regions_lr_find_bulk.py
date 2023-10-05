import subprocess

dry_run = False

regions = '0-8'
regions = '0'
seeds = [42, 43, 44, 45, 46]
seeds = [42]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_pmcc_2_regions_lr_find.slurm'

for seed in seeds:
    export = f'ALL,RANDOM_SEED={seed}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    if not dry_run:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
