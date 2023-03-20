#!/bin/bash

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-444"
REGION="Brain"
MODEL_NAME="localiser-replan-$REGION"
RUN_NAME="single-class"
N_EPOCHS=200
N_GPUS=1
N_NODES=1
N_WORKERS=8
PRETRAINED=None
RESUME=False
RESUME_CHECKPOINT="last"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser/train_localiser.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --n_epochs $n_EPOCHS \
    --n_gpus $n_GPUS \
    --n_nodes $n_NODES \
    --n_subset $n_SUBSET \
    --n_workers $n_WORKERS \
    --pretrained $PRETRAINED \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_job_id $SLURM_JOB_ID \
    --use_logger $USE_LOGGER
