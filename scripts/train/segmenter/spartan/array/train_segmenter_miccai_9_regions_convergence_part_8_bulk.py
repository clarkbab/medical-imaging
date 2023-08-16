import subprocess

factors = [1, 2, 4, 8, 16, 32, 64, 128]
random_seeds = [42, 43, 44]
# 'sbatch' needs quotations around value otherwise it will split on ','.
cvg_thresholds = "\"[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]\""
resume = False
complexity_weights = True
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence_part_8.slurm'

for factor in factors:
    for random_seed in random_seeds:
        run_name = f"factor-{factor}-seed-{random_seed}"
        export = f'ALL,COMPLEXITY_WEIGHTS_FACTOR={factor},DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS={cvg_thresholds},RANDOM_SEED={random_seed},RESUME={resume},RUN_NAME={run_name},USE_COMPLEXITY_WEIGHTS={complexity_weights}'
        command = f'sbatch --array=0 --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
