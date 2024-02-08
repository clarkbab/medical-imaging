#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
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

# Get model name.
RESOLUTION="222"
DATASET="PMCC-HN-REPLAN-$RESOLUTION"
MODEL_NAME="segmenter-replan-$RESOLUTION"

# Set up region weighting.
USE_WEIGHTS=True
WEIGHTS_IV_FACTOR_0=0.5
SCHEDULE_EPOCHS=200

# Get run name.
N_FOLDS=5
TEST_FOLD=$SLURM_ARRAY_TASK_ID
RUN_NAME="n-folds-$N_FOLDS-fold-$TEST_FOLD"
if [[ $USE_WEIGHTS == True ]]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
if [[ $TEST_FOLD -eq 0 ]] || [[ $TEST_FOLD -eq 4 ]]; then
    RANDOM_SEED=43
else
    RANDOM_SEED=42
fi
RUN_NAME="$RUN_NAME-seed-$RANDOM_SEED"

# Loader settings.
LOAD_ALL_SAMPLES=True
N_FOLDS=5
SHUFFLE_SAMPLES=False
USE_GROUPING=True
USE_SPLIT_FILE=False

# Other settings.
CHECK_EPOCHS=True
CROP_MM="(330,380,500)"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
N_EPOCHS=1000
REGION='RL:PMCC_REPLAN'
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
    --shuffle_samples $SHUFFLE_SAMPLES \
    --test_fold $TEST_FOLD \
    --use_grouping $USE_GROUPING \
    --use_split_file $USE_SPLIT_FILE
