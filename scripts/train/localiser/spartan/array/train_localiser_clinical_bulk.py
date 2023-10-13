import subprocess

dry_run = True

regions = '0-16'
script = 'scripts/train/localiser/spartan/array/train_localiser_clinical.slurm'
regionses = ['3', '4-7,9,10', '0-2,8,11-14', '15,16']
regionses = ['3', '4-7,9,10']
test_folds = [0, 1, 2, 3, 4]
test_folds = [0, 1, 2, 3, 4]
n_trains = [5, 10, 20, 50, 100, 200, None]
n_trains = [5, 10, 20]
resume = False

n_train_epochses = {
    '3': {
        5: 1200,
        10: 600,
        20: 450,
    },
    '4-7,9,10': {
        5: 1800,
        10: 900,
        20: 600,
        50: 300,
    },
    '0-2,8,11-16': {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    '15,16': {
        5: ,
        10: 900,
        20: 600,
        50: 300
    }
}

for regions in regionses:
    for n_train in n_trains:
        for test_fold in test_folds:
            # Get number of epochs.
            n_train_epochs = n_train_epochses[regions]
            n_epochs = n_train_epochs[n_train] if n_train in n_train_epochs else n_train_epochs['default']

            # Create slurm command.
            export = f'ALL,N_EPOCHS={n_epochs},N_TRAIN={n_train},RESUME={resume},TEST_FOLD={test_fold}'
            command = f'sbatch --array={regions} --export={export} {script}' 
            print(command)
            if not dry_run:
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.communicate()
