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
EXCLUDE_LIKE="'-0'"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
REGION='RL:PMCC_REPLAN'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --exclude_like $EXCLUDE_LIKE \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --shuffle_samples $SHUFFLE_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_grouping $USE_GROUPING \
    --use_split_file $USE_SPLIT_FILE
