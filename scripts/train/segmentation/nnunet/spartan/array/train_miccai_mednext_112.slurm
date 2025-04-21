#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set nnUNet vars.
export nnUNet_raw_data_base="/data/projects/punim1413/mymi/datasets/nnunet/v1/raw"
# export nnUNet_predictions="/data/projects/punim1413/mymi/datasets/nnunet/v1/predictions"
export nnUNet_preprocessed="/data/projects/punim1413/mymi/datasets/nnunet/v1/preprocessed"
export RESULTS_FOLDER="/data/projects/punim1413/mymi/models/nnunet/v1"

REGIONS=(
    'Bone_Mandible'     # 0
    'Brainstem'         # 1
    'Glnd_Submand_L'    # 2
    'Glnd_Submand_R'    # 3
    'OpticChiasm'       # 4
    'OpticNrv_L'        # 5
    'OpticNrv_R'        # 6
    'Parotid_L'         # 7
    'Parotid_R'         # 8
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
FIRST_TASK_ID=001
TASK_ID=$(($FIRST_TASK_ID + $SLURM_ARRAY_TASK_ID))

MODEL="3d_fullres"
TRAINER="nnUNetTrainerV2_MedNeXt_B_kernel3"
FOLD="all"
PLAN="nnUNetPlansv2.1"

# --c restarts from latest checkpoint.
command="mednextv1_train
    $MODEL \
    $TRAINER \
    $TASK_ID \
    $FOLD \
    -p $PLAN"
echo $command
eval $command
