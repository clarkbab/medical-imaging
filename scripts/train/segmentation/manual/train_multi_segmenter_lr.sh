#!/bin/bash

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="MICCAI-2015-222"
# REGION="('Bone_Mandible','BrachialPlex_L','Brain','Lens_L','Parotid_L')"
REGION="['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R']"
REGION_SHORT="BM"
MODEL_NAME="segmenter-miccai"
RESOLUTION="222"
RUN_NAME="baseline-2-regions-$REGION_SHORT-$RESOLUTION"
LR_FIND=True
LR_INIT=1e-3
LR_SCHEDULER=False
N_EPOCHS=500
N_GPUS=1
N_NODES=1
N_SPLIT_CHANNELS=2
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True

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
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER

