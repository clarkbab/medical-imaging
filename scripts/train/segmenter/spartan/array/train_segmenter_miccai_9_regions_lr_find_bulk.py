import subprocess

resolutions = ['112', '222', '444']
resolutions = ['444']
random_seeds = [42, 43, 44, 45, 46]
random_seeds = [42]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_lr_find.slurm'

for resolution in resolutions:
    for random_seed in random_seeds:
        export = f'ALL,RESOLUTION={resolution},RANDOM_SEED={random_seed}'
        command = f'sbatch --array=0 --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
