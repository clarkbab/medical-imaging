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

DATASET_ID=115
CONFIGURATION="3d_fullres"
echo $DATASET_ID

nnUNetv2_plan_and_preprocess \
    -c $CONFIGURATION \
    -d $DATASET_ID \
    -gpu_memory_target 60 \
    -overwrite_plans_name nnUNetResEncUNetPlansXXL \
    -pl nnUNetPlannerResEncXL \
    --verify_dataset_integrity
