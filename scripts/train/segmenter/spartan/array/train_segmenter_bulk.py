import subprocess

regions = '2'
script = 'scripts/train/segmenter/spartan/array/train_segmenter.slurm'
test_folds = [0, 1, 2, 3, 4]
test_folds = [0]
models = ['clinical']
n_trains = [5, 10, 20, 50, 100, None]
n_trains = [None]
resume = True

n_train_epochs = {
    5: 900,             # BP_L/R @ n=5 took this long to plateau.
    10: 450,            # BP_L/R, L_L/R @ n=10.
    20: 300,            # BP_L/R, ON_L/R @ n=20.
    'default': 150      # All other models.
}

for model in models:
    for test_fold in test_folds:
        for n_train in n_trains:
            # Get number of epochs.
            n_epochs = n_train_epochs[n_train] if n_train in n_train_epochs else n_train_epochs['default']

            # Create slurm command.
            export = f'ALL,MODEL={model},N_EPOCHS={n_epochs},N_TRAIN={n_train},RESUME={resume},TEST_FOLD={test_fold}'
            command = f'sbatch --array={regions} --export={export} {script}' 
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
