import subprocess

regions = '0-16'
script = 'scripts/predict/nifti/localiser/spartan/array/predict_localiser_public.slurm'
test_folds = [0, 1, 2, 3, 4]

for test_fold in test_folds:
    # Create slurm command.
    export = f'ALL,TEST_FOLD={test_fold}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
