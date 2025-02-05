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

# Get model name.
RESOLUTION="112"
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
RANDOM_SEED=42
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
EXCLUDE_LIKE=None
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
N_EPOCHS=1000
REGION='RL:PMCC_REPLAN'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_TEST_LOADER=False

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi_nnunet_bootstrap.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --exclude_like $EXCLUDE_LIKE \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --shuffle_samples $SHUFFLE_SAMPLES \
    --test_fold $TEST_FOLD \
    --use_grouping $USE_GROUPING \
    --use_split_file $USE_SPLIT_FILE \
    --use_test_loader $USE_TEST_LOADER
