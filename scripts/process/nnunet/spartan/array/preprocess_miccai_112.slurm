#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set nnUNet vars.
export nnUNet_raw="/data/projects/punim1413/mymi/datasets/nnunet/raw"
export nnUNet_predictions="/data/projects/punim1413/mymi/datasets/nnunet/predictions"
export nnUNet_preprocessed="/data/projects/punim1413/mymi/datasets/nnunet/preprocessed"
export nnUNet_results="/data/projects/punim1413/mymi/models/nnunet"

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
CONFIGURATION="3d_fullres"
echo $DATASET_ID

nnUNetv2_plan_and_preprocess \
    -c $CONFIGURATION \
    -d $DATASET_ID \
    -gpu_memory_target 80 \
    -overwrite_plans_name nnUNetResEncUNetPlansXXL \
    -pl nnUNetPlannerResEncXL \
    --verify_dataset_integrity
