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
export nnUNet_raw="/data/projects/punim1413/mymi/datasets/nnunet/raw"
export nnUNet_preprocessed="/data/projects/punim1413/mymi/datasets/nnunet/preprocessed"
export nnUNet_results="/data/projects/punim1413/mymi/models/nnunet"
export nnUNet_keep_files_open=True

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

FIRST_DATASET_ID=106
DATASET_ID=$(($FIRST_DATASET_ID + $SLURM_ARRAY_TASK_ID))
echo $DATASET_ID

MODEL="3d_fullres"
FOLD="all"
PLAN="nnUNetResEncUNetPlansXXL"
TRAINER="nnUNetTrainerNoMirroring"

# --c restarts from latest checkpoint.
command="nnUNetv2_train
    $DATASET_ID \
    $MODEL \
    $FOLD \
    -p $PLAN \
    -tr $TRAINER"
echo $command
eval $command
