#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-01:00:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load fosscuda/2020b
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['PMCC-HN-TEST-SEG','PMCC-HN-TRAIN-SEG']"
MODEL_NAME="segmenter-multi"
RUN_NAME="test-slurm"
N_EPOCHS=150
N_FOLDS=5
N_GPUS=1
N_NODES=1
N_TRAIN=None
N_WORKERS=8
PRETRAINED_MODEL=None
RESUME=False
RESUME_RUN=None
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOGGER=False

export CUDA_LAUNCH_BLOCKING=1

python $SCRIPT_DIR/train/segmenter/train_multi.py \
    --dataset $DATASETS \
    --model $MODEL_NAME \
    --run $RUN_NAME \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --pretrained_model $PRETRAINED_MODEL \
    --use_logger $USE_LOGGER \
    --resume $RESUME \
    --resume_run $RESUME_RUN \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD
