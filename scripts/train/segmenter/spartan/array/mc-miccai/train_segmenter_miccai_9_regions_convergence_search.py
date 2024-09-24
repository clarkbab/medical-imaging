import subprocess

dry_run = False

script = 'scripts/train/segmenter/spartan/array/mc-miccai/train_segmenter_miccai_9_regions_convergence_search.slurm'

lams = [0, 0.1, 0.3, 0.5]
lrs = [1e-4, 5e-4, 1e-3]
opts = ['adam', 'rmsprop', 'sgd']

for lam in lams:
    for lr in lrs:
        for opt in opts:
            export = f'ALL,LAM={lam},LR_INIT={lr},OPTIMISER={opt}'
            command = f'sbatch --export={export} {script}' 
            print(command)
            if not dry_run:
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.communicate()
