#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-06:00:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
REGION='Brain'
OUTPUT_SPACING="(4,4,4)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TRAINING_DATASET="$DATASET-LOC"

python $SCRIPT_DIR/process/nifti/convert_to_training.py \
    --dataset $DATASET \
    --region $REGION \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET
