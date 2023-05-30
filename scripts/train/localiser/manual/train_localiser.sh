#!/bin/bash

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-444"
REGION="Brain"
MODEL_NAME="localiser-$REGION"
RUN_NAME="single-class"
N_EPOCHS=200
N_GPUS=1
N_NODES=1
N_WORKERS=8
PRETRAINED=None
RESUME=False
RESUME_CKPT="last"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser/train_localiser.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --region $REGION \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_subset $N_SUBSET \
    --n_workers $N_WORKERS \
    --pretrained $PRETRAINED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
