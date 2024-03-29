#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 64G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 7-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

# Set nnUNet vars.
export nnUNet_raw_data_base='/data/gpfs/projects/punim1413/nnunet/data'
export nnUNet_preprocessed='/data/gpfs/projects/punim1413/nnunet/data/nnUNet_preprocessed'
export RESULTS_FOLDER='/data/gpfs/projects/punim1413/nnunet/data/nnUNet_models'

TASKS=(
    'Task500_Brain'
    'Task501_Lens_L'
    'Task502_Parotid_L'
)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
# CONF=2d
# FOLD=0
# RESUME='-c'

# Set 'TRAINER' var.
case "$CONF" in
    '3d_cascade_fullres')
        TRAINER='nnUNetTrainerV2CascadeFullRes';;
    *)
        TRAINER='nnUNetTrainerV2';;
esac

# Set 'RESUME' var.
case "$RESUME" in
    'True')
        RESUME='-c';;
    'False')
        RESUME='';;
esac

command="nnUNet_train $CONF $TRAINER $TASK $FOLD --npz $RESUME"
echo $command
eval $command
