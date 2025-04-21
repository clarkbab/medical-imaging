#! /usr/bin/env bash
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
python --version

DATASETS="['PMCC-HN-TEST-SEG','PMCC-HN-TRAIN-SEG']"
REGION="Brain"
MODEL_NAME="segmenter-$REGION"
N_EPOCHS=5
N_GPUS=1
N_NODES=1
N_WORKERS=1
N_TRAIN=None
PRETRAINED_MODEL=None
RESUME=False
RESUME_CKPT=None
RUN_NAME="test-sharded"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOGGER=False

python $SCRIPT_DIR/train/segmenter/train.py \
    --datasets $DATASETS \
    --region $REGION \
    --model $MODEL_NAME \
    --run $RUN_NAME \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --pretrained_model $PRETRAINED_MODEL \
    --use_logger $USE_LOGGER \
    --region $REGION \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD
