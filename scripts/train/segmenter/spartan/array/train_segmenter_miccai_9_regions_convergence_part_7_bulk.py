import subprocess

factors = [1, 2, 4, 8, 16, 32, 64, 128]
random_seeds = [42, 43, 44]
# Still require 'thresholds' for calculating convergence.
cvg_thresholds = "\"[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]\""
resume = False
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence_part_7.slurm'

for factor in factors:
    for random_seed in random_seeds:
        run_name = f"factor-{factor}-seed-{random_seed}"
        export = f'ALL,DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS={cvg_thresholds},RANDOM_SEED={random_seed},RESUME={resume},RUN_NAME={run_name},WEIGHTING_FACTOR={factor}'
        command = f'sbatch --array=0 --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
