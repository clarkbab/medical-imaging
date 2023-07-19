import subprocess

weighting_schemes = ['1a', '2a', '2b', '3a', '4a', '4b', '4c', '4d']
weighting_schemes = ['2b']
random_seeds = [45, 46]
n_epochs = 10000
lr_init = 1e-3
resume = False
script = 'scripts/train/segmenter/spartan/array/train_segmenter_miccai_9_regions_convergence.slurm'

for weighting_scheme in weighting_schemes:
    for random_seed in random_seeds:
        run_name = f"9-regions-ALL-scheme-{weighting_scheme}-seed-{random_seed}"
        resume_run = f"9-regions-ALL-scheme-{weighting_scheme}-seed-{random_seed}-epochs-40k"
        if lr_init != 1e-3:
            lr_init_str = str(lr_init)
            run_name = f"{run_name}-lr-{lr_init_str}"
        if n_epochs != 10000:
            n_epochs_str = str(int(n_epochs / 1000)) + 'k'
            run_name = f"{run_name}-epochs-{n_epochs_str}"

        export = f'ALL,LR_INIT={lr_init},N_EPOCHS={n_epochs},RANDOM_SEED={random_seed},RESUME={resume},RESUME_RUN={resume_run},RUN_NAME={run_name},WEIGHTING_SCHEME={weighting_scheme}'
        command = f'sbatch --array=0 --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
