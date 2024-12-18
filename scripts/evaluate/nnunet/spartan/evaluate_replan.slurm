#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
##SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
DATASET="PMCC-HN-REPLAN"
EXCLUDE_LIKE="\"-0\""
# REGION="Brain"
# FOLD=0

python $SCRIPT_DIR/evaluate/nnunet/evaluate.py \
    --dataset $DATASET \
    --region $REGION \
    --fold $FOLD \
    --exclude_like $EXCLUDE_LIKE
