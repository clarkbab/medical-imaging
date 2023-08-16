import numpy as np
import subprocess

dry_run = True
factors = [1, 2, 5, 10, 20, 50]
random_seeds = [42, 43, 44]
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence_volume.slurm'

inv_volumes = np.array([
    1.81605095e-05,
    3.75567497e-05,
    1.35979999e-04,
    1.34588032e-04,
    1.71684281e-03,
    1.44678695e-03,
    1.63991258e-03,
    3.45656440e-05,
    3.38292316e-05
])

for factor in factors:
    # Calculate weights.
    weights = inv_volumes ** (1 / factor)
    weights = np.array(weights) / np.sum(weights)
    weights = f"\"{ ','.join(str(weights).split()) }\""

    for random_seed in random_seeds:
        run_name = f"seed-{random_seed}"
        export = f'ALL,RANDOM_SEED={random_seed},RUN_NAME={run_name},WEIGHTS={weights}'
        command = f'sbatch --array=0 --export={export} {script}' 
        print(command)
        if not dry_run:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
