#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
#SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
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
MODEL_NAME="segmenter-replan-eyes-v3"
RESOLUTION="112"
DATASET="PMCC-HN-REPLAN-EYES-$RESOLUTION"
WEIGHTS=None
WEIGHTS_IV_FACTOR_0=1.0
WEIGHTS_IV_FACTOR="[$WEIGHTS_IV_FACTOR_0,0]"
SCHEDULE_EPOCHS=200
RESUME=False
LR_INIT=1e-4
USE_WEIGHTS=False

# Get run name.
RANDOM_SEED=44
RUN_NAME="2-regions-$SHORT_REGION"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
if [ "$LR_INIT" = 1e-4 ]; then
    RUN_NAME="$RUN_NAME-lr-$LR_INIT"
fi
RUN_NAME="$RUN_NAME-seed-$RANDOM_SEED"

# Add augmentation.
ROTATION="(45,5,5)"
SCALE="(0.8,1.2)"
TRANSLATION="(-50,50,-50,50,70,170)"

# Other params.
BATCH_SIZE=1
CKPT_MODEL=True
N_EPOCHS=2000
N_GPUS=1
N_NODES=1
N_SPLIT_CHANNELS=2
N_WORKERS=8
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0
USE_LOGGER=True
WEIGHTS_SCHEDULE="[0,$SCHEDULE_EPOCHS]"

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --rotation $ROTATION \
    --scale $SCALE \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --translation $TRANSLATION \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_weights $USE_WEIGHTS \
    --weights $WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR \
    --weights_schedule $WEIGHTS_SCHEDULE
