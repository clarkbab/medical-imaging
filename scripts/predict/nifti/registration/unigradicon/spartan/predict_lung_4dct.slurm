#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
##SBATCH --mem 128G
#SBATCH --mem 120G
#SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-01:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="DIRLAB-LUNG-4DCT"
MODEL="unigradicon"
REGISTER_CT=True
LANDMARKS="all"
REGIONS="all"

command="python $MYMI_CODE/scripts/predict/nifti/registration/unigradicon/predict.py \
    --dataset $DATASET \
    --model $MODEL \
    --register_ct $REGISTER_CT \
    --landmarks $LANDMARKS \
    --regions $REGIONS"
echo $command
eval $command
