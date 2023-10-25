import subprocess

dry_run = False

script = 'scripts/predict/nifti/localiser/spartan/array/predict_localiser_clinical.slurm'
regions = list(range(17))
regions = [14]
test_folds = list(range(5))
test_folds = [1]
n_trains = [5, 10, 20, 50, 100, 200, None]
n_trains = [None]

n_train_epochses = {
    0: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    1: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    2: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    3: {
        5: 1200,
        10: 600,
        20: 450,
        'default': 150
    },
    4: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
    5: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
    6: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
    7: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
    8: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    9: {
        5: 1800,
        10: 900,
        20: 600,
        50: 300,
        'default': 150
    },
    10: {
        5: 1800,
        10: 900,
        20: 600,
        50: 300,
        'default': 150
    },
    11: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    12: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    13: {
        5: 900,             # BP_L/R @ n=5 took this long to plateau.
        10: 450,            # BP_L/R, L_L/R @ n=10.
        20: 300,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    14: {
        5: 1350,             # BP_L/R @ n=5 took this long to plateau.
        10: 700,            # BP_L/R, L_L/R @ n=10.
        20: 450,            # BP_L/R, ON_L/R @ n=20.
        'default': 150      # All other models.
    },
    15: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
    16: {
        5: 2500,
        10: 1350,
        20: 600,
        50: 300,
        'default': 150
    },
}

for region in regions:
    for n_train in n_trains:
        for test_fold in test_folds:
            # Get number of epochs.
            n_train_epochs = n_train_epochses[region]
            n_epochs = n_train_epochs[n_train] if n_train in n_train_epochs else n_train_epochs['default']

            # Create slurm command.
            export = f'ALL,N_EPOCHS={n_epochs},N_TRAIN={n_train},TEST_FOLD={test_fold}'
            command = f'sbatch --array={region} --export={export} {script}' 
            print(command)
            if not dry_run:
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                process.communicate()
