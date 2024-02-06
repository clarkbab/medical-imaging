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

REGIONS=(
    "['Eye_L','Eye_R']"     # 0
)
SHORT_REGIONS=(
    'EL_ER'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}

# Get model name.
MODEL_NAME="segmenter-replan-eyes-v3"
LR_INIT=1e-4
RANDOM_SEED=43
SCHEDULE_EPOCHS=200
USE_WEIGHTS=False
WEIGHTS_IV_FACTOR_0=1.0
RANDOM_SEED=43
RUN_NAME="2-regions-$SHORT_REGION"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
if [ "$LR_INIT" = 1e-4 ]; then
    RUN_NAME="$RUN_NAME-lr-$LR_INIT"
fi
RUN_NAME="$RUN_NAME-seed-$RANDOM_SEED"

# Loader settings.
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False

# Other settings.
DATASET="PMCC-HN-REPLAN-EYES-UNIFORM"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
MODEL_REGION="['Eye_L','Eye_R']"
MODEL_SPACING="(1,1,2)"
CHECK_EPOCHS=True
CROP_MM="(330,380,500)"
N_EPOCHS=1000
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/predict/nifti/segmenter/predict_all_multi.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --model_spacing $MODEL_SPACING \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --n_epochs $N_EPOCHS \
    --region $REGION
