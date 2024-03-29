import subprocess

tasks = '0,1,2'
script = 'scripts/train/nnunet/spartan/array/train_nnunet.slurm'
confs = ['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres']
confs = ['3d_cascade_fullres']
folds = list(range(5))
resume = False

for conf in confs:
    for fold in folds:
        # Create slurm command.
        export = f'ALL,CONF={conf},FOLD={fold},RESUME={resume}'
        command = f'sbatch --array={tasks} --export={export} {script}' 
        print(command)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        process.communicate()
