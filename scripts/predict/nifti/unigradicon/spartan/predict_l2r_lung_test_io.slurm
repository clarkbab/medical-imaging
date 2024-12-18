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
#SBATCH --time 0-01:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="L2R-LUNG-CT-TEST"
MODELNAME="UNIGRADICON-IO"
REGISTER_IMAGES=True
REGION="Lung"

command="python $MYMI_CODE/scripts/predict/nifti/unigradicon/predict_io.py \
    --dataset $DATASET \
    --model $MODEL \
    --modelname $MODELNAME \
    --register_images $REGISTER_IMAGES \
    --region $REGION"
echo $command
eval $command
