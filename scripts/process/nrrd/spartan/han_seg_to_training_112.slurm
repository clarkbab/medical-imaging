#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 120G
##SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
DATASET="HAN-SEG"
REGIONS=None    # Defaults to all regions.
DILATE_ITER=0
DILATE_REGIONS="()"
OUTPUT_SPACING="(1,1,2)"
TRAINING_DATASET="$DATASET-112"

python $SCRIPT_DIR/process/nrrd/convert_to_training.py \
    --dataset $DATASET \
    --dilate_iter $DILATE_ITER \
    --dilate_regions $DILATE_REGIONS \
    --regions $REGIONS \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET
