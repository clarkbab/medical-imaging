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
RESOLUTION="222"
DATASET="PMCC-HN-REPLAN-POP-$RESOLUTION"
MODEL_NAME="segmenter-replan-pop-$RESOLUTION"

# Set up region weighting.
USE_WEIGHTS=True
WEIGHTS_IV_FACTOR_0=0.5
SCHEDULE_EPOCHS=200

# Get run name.
RANDOM_SEED=42
TEST_FOLD=$SLURM_ARRAY_TASK_ID
RUN_NAME_BASE="17-regions-fold-$TEST_FOLD"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME_BASE-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS-seed-$RANDOM_SEED"
else
    RUN_NAME="$RUN_NAME_BASE-seed-$RANDOM_SEED"
fi

# Loader settings.
LOAD_ALL_SAMPLES=True
LOADER_SHUFFLE_SAMPLES=False
N_FOLDS=5
USE_LOADER_GROUPING=True
USE_LOADER_SPLIT_FILE=False

# Other settings.
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
REGION='RL:PMCC_REPLAN'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --loader_shuffle_samples $LOADER_SHUFFLE_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_grouping $USE_LOADER_GROUPING \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
