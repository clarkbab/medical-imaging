import subprocess

regions = '0-16'
script = 'scripts/evaluate/nifti/localiser/spartan/array/evaluate_localiser.slurm'
test_folds = [0, 1, 2, 3, 4]

for test_fold in test_folds:
    # Create slurm command.
    export = f'ALL,TEST_FOLD={test_fold}'
    command = f'sbatch --array={regions} --export={export} {script}' 
    print(command)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    process.communicate()
