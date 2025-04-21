import subprocess

script = 'scripts/train/segmenter/nnunet/spartan/array/train_ref.slurm'
# confs = ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']
confs = ['3d_cascade_fullres']
folds = list(range(5))
folds = [2]
regions = '0,1'
resume = True

for conf in confs:
    for fold in folds:
        # Create slurm command.
        export = f'ALL,CONF={conf},TEST_FOLD={fold},RESUME={resume}'
        command = f'sbatch --array={regions} --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
