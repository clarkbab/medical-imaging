#!/bin/bash

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
# REGION="('Bone_Mandible','BrachialPlex_L','Brain','Lens_L','Parotid_L')"
REGION="OpticChiasm"
REGION_SHORT="OC"
MODEL_NAME="segmenter-miccai-no-background"
RUN_NAME="1-region-$REGION_SHORT-$RESOLUTION"
LR_FIND=False
LR_INIT=1e-4
LR_SCHEDULER=False
N_EPOCHS=10000
N_GPUS=1
N_NODES=1
N_SPLIT_CHANNELS=2
N_WORKERS=8
PRECISION='bf16'
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=False

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --halve_channels $HALVE_CHANNELS \
    --lr_find $LR_FIND \
    --lr_init $LR_INIT \
    --lr_scheduler $LR_SCHEDULER \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --precision $PRECISION \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER

