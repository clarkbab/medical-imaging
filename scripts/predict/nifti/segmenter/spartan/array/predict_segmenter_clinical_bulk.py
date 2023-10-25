import subprocess

script = 'scripts/predict/nifti/segmenter/spartan/array/predict_segmenter_clinical.slurm'
regions = list(range(17))
regions = [2]
test_folds = list(range(5))
n_trains = [5, 10, 20, 50, 100, 200, None]

for region in regions:
    for test_fold in test_folds:
        for n_train in n_trains:
            # Create slurm command.
            export = f'ALL,N_TRAIN={n_train},TEST_FOLD={test_fold}'
            command = f'sbatch --array={region} --export={export} {script}' 
            print(command)
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
