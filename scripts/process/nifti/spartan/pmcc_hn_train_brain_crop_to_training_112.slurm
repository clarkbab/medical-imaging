#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition cascade
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 64G
##SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

# Set resolution.
SPACING="(1,1,2)"
RESOLUTION="112"
DEST_DATASET="PMCC-HN-TRAIN-BC-$RESOLUTION"

# Set other parameters.
DATASET="PMCC-HN-TRAIN"
CROP_MM="(300,400,500)"
REGION="RL:PMCC"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nifti/convert_brain_crop_to_training.py \
    --dataset $DATASET \
    --region $REGION \
    --dest_dataset $DEST_DATASET \
    --crop_mm $CROP_MM \
    --spacing $SPACING