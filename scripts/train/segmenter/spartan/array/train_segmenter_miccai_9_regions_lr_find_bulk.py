import subprocess

weighting_schemes = ['5a', '5b', '5c', '5d']
resolutions = ['112', '222', '444']
precisions = [32, 'bf16']
random_seeds = [42, 43, 44, 45, 46]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_lr_find.slurm'

for weighting_scheme in weighting_schemes:
    for resolution in resolutions:
        for precision in precisions:
            for random_seed in random_seeds:
                export = f'ALL,RESOLUTION={resolution},PRECISION={precision},RANDOM_SEED={random_seed},WEIGHTING_SCHEME={weighting_scheme}'
                command = f'sbatch --array=0 --export={export} {script}' 
                print(command)
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.communicate()
