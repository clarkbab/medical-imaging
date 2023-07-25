import subprocess

factors = [0.01, 0.1, 0.2, 0.4, 0.8]
delays = [10, 50, 100, 200, 500, 1000]
random_seeds = [42, 43, 44]
# 'sbatch' needs quotations around value otherwise it will split on ','.
cvg_thresholds = "\"[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]\""
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence.slurm'

for factor in factors:
    for delay in delays:
        for random_seed in random_seeds:
            run_name = f"factor-{factor}-delay-{delay}-seed-{random_seed}"
            export = f'ALL,DYNAMIC_WEIGHTS_CONVERGENCE_DELAY={delay},DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS={cvg_thresholds},DYNAMIC_WEIGHTS_FACTOR={factor},RANDOM_SEED={random_seed},RUN_NAME={run_name}'
            command = f'sbatch --array=0 --export={export} {script}' 
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
