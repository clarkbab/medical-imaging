#! /usr/bin/env bash
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
python --version

DATASETS="['PMCC-HN-TEST-MULTI','PMCC-HN-TRAIN-MULTI']"
MODEL_NAME="segmenter-multi"
N_EPOCHS=1
N_GPUS=1
N_NODES=1
N_WORKERS=1
N_TRAIN=None
REGIONS="['Brain','Lens_L','OpticNerve_L','Parotid_L','Submandibular_L']"
RESUME=False
RESUME_CKPT=None
RUN_NAME="gpu-test"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOGGER=False

python $SCRIPT_DIR/train/segmenter/train_memory_test.py \
    --dataset $DATASETS \
    --model $MODEL_NAME \
    --run $RUN_NAME \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --regions $REGIONS \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
