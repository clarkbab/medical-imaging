import subprocess

dry_run = False

regions = '16'
script = 'scripts/evaluate/nifti/two-stage/spartan/array/evaluate_two_stage_hanseg.slurm'
test_folds = [0, 1, 2, 3, 4]
test_folds = [1, 4]

for test_fold in test_folds:
        # Create slurm command.
        export = f'ALL,TEST_FOLD={test_fold}'
        command = f'sbatch --array={regions} --export={export} {script}' 
        print(command)
        if not dry_run:
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            process.communicate()
