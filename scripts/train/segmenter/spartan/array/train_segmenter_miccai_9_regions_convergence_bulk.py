import subprocess

factors = [1, 2, 5, 10, 20]
factors = [20]
delays = [20, 50, 100]
delays = [20]
delays_2 = [5] * len(delays)
random_seeds = [42, 43, 44]
random_seeds = [42]
# 'sbatch' needs quotations around value otherwise it will split on ','.
cvg_thresholds = "\"[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]\""
resume = False
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence.slurm'

for factor in factors:
    for delay, delay_2 in zip(delays, delays_2):
        for random_seed in random_seeds:
            run_name = f"factor-{factor}-delay-{delay}-seed-{random_seed}"
            export = f'ALL,DYNAMIC_WEIGHTS_CONVERGENCE_DELAY={delay},DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS={cvg_thresholds},DYNAMIC_WEIGHTS_FACTOR={factor},RANDOM_SEED={random_seed},RESUME={resume},RUN_NAME={run_name}'
            command = f'sbatch --array=0 --export={export} {script}' 
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
