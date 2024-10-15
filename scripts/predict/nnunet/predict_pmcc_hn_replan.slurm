#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
##SBATCH --mem 120G
#SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 1-00:00:00
##SBATCH --time 0-04:00:00
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

DATASET="PMCC-HN-REPLAN"
# NNUNET_DATASET="011"
# FOLD=0

nnUNetv2_predict \
    -i $nnUNet_raw/Dataset${NNUNET_DATASET}_REF_MODEL_FOLD_$FOLD/imagesTs \
    -o $nnUNet_predictions/fold-$FOLD \
    -d $NNUNET_DATASET \
    -p nnUNetResEncUNetPlansXXL \
    -c 3d_fullres \
    -f all
