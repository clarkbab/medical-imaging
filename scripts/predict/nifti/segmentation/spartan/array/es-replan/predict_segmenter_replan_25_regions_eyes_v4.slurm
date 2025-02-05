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

REGION="RL:PMCC_REPLAN_ALL"
MODEL_NAME="segmenter-replan-eyes-v4"
RESOLUTION="222"
DATASET="PMCC-HN-REPLAN-ALL-$RESOLUTION"
WEIGHTS=None
WEIGHTS_IV_FACTOR_0=1.1
WEIGHTS_IV_FACTOR="[$WEIGHTS_IV_FACTOR_0,0]"
SCHEDULE_EPOCHS=200
RESUME=False
LR_INIT=1e-3
USE_WEIGHTS=False

# Get run name.
RANDOM_SEED=42
RUN_NAME="25-regions"
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
CHECK_EPOCHS=True
CROP_MM="(330,380,500)"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
N_EPOCHS=2000
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
