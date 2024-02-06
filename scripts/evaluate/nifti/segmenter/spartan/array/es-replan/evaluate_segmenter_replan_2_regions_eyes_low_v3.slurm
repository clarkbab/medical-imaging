#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Get region names.
REGIONS=(
    "['Eye_L','Eye_R']"     # 0
)
SHORT_REGIONS=(
    'EL_ER'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}

# Get model name.
RESOLUTION="112"
MODEL_NAME="segmenter-replan-eyes-v3"
LR_INIT=1e-4
RANDOM_SEED=43
SCHEDULE_EPOCHS=200
USE_WEIGHTS=False
WEIGHTS_IV_FACTOR_0=1.0
RUN_NAME="2-regions-$SHORT_REGION"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
if [ "$LR_INIT" = 1e-4 ]; then
    RUN_NAME="$RUN_NAME-lr-$LR_INIT"
fi
RUN_NAME="$RUN_NAME-seed-$RANDOM_SEED"

# Other settings.
DATASET="PMCC-HN-REPLAN-EYES-LOW"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_all_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL
