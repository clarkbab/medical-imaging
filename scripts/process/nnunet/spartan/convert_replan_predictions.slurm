#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
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

DATASET="PMCC-HN-REPLAN"
# REGION="Brain"
# FOLD=0
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nnunet/convert_replan_predictions.py \
    --dataset $DATASET \
    --region $REGION \
    --fold $FOLD
