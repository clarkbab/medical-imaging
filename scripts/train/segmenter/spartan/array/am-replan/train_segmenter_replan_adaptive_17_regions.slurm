#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 128G
##SBATCH --mem 120G
#SBATCH --cpus-per-gpu 16
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Get model name.
RESOLUTION="222"
DATASET="PMCC-HN-REPLAN-ADPTM-$RESOLUTION"
MODEL_NAME="segmenter-replan-$RESOLUTION"

# Set up region weighting.
USE_WEIGHTS=True
WEIGHTS_IV_FACTOR_0=0.5
WEIGHTS_IV_FACTOR="[$WEIGHTS_IV_FACTOR_0,0]"
SCHEDULE_EPOCHS=200
WEIGHTS_SCHEDULE="[0,$SCHEDULE_EPOCHS]"

# Get run name.
N_FOLDS=5
TEST_FOLD=$SLURM_ARRAY_TASK_ID
RUN_NAME="adaptive-n-folds-$N_FOLDS-fold-$TEST_FOLD"
RANDOM_SEED=42
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
RUN_NAME="$RUN_NAME-seed-$RANDOM_SEED"

# Loader settings.
LOAD_ALL_SAMPLES=True
LOADER_SHUFFLE_SAMPLES=False
USE_LOADER_GROUPING=True

# Other settings.
LR_INIT=1e-3
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=16
REGION='RL:PMCC_REPLAN'
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_GROUPING=True
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter/train_adaptive_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --loader_load_all_samples $LOADER_LOAD_ALL_SAMPLES \
    --loader_shuffle_samples $LOADER_SHUFFLE_SAMPLES \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_loader_grouping $USE_LOADER_GROUPING \
    --use_logger $USE_LOGGER \
    --use_weights $USE_WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR \
    --weights_schedule $WEIGHTS_SCHEDULE
