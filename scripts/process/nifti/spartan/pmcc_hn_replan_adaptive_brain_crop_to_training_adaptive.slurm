#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

# Set resolution.
SPACING="(2,2,2)"
RESOLUTION="222"
DEST_DATASET="PMCC-HN-REPLAN-ADPT-$RESOLUTION"

# Set other parameters.
DATASET="PMCC-HN-REPLAN"
CROP_MM="(330,380,500)"
REGION="RL:PMCC_REPLAN"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nifti/convert_adaptive_brain_crop_to_training_adaptive.py \
    --dataset $DATASET \
    --region $REGION \
    --dest_dataset $DEST_DATASET \
    --crop_mm $CROP_MM \
    --spacing $SPACING
