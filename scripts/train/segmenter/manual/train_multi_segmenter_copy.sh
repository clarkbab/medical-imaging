#!/bin/bash

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGION="['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-arch-modification"
RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
RANDOM_SEED=42
USE_CVG_WEIGHTING=True
WEIGHTS_EPOCH=1000
CW_CVG_DELAY_ABOVE=20
CW_CVG_DELAY_BELOW=5
CW_CVG_THRESHOLDS="[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]"
CW_FACTOR_0=10
CW_FACTOR="[$CW_FACTOR_0,0]"
CW_SCHEDULE="[0,$WEIGHTS_EPOCH]"
USE_WEIGHTS=True
WEIGHTS=None
WEIGHTS_IV_FACTOR_0=1
WEIGHTS_IV_FACTOR="[$WEIGHTS_IV_FACTOR_0,0]"
WEIGHTS_SCHEDULE="[0,$WEIGHTS_EPOCH]"
N_SPLIT_CHANNELS=2
RUN_NAME="channels-$N_SPLIT_CHANNELS-seed-$RANDOM_SEED-cw-$CW_FACTOR_0-ivw-$WEIGHTS_IV_FACTOR_0"
BATCH_SIZE=1
CKPT_MODEL=True
LR_INIT=1e-3
N_EPOCHS=5000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True
USE_LR_SCHEDULER=False

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --cw_cvg_delay_above $CW_CVG_DELAY_ABOVE \
    --cw_cvg_delay_below $CW_CVG_DELAY_BELOW \
    --cw_cvg_thresholds $CW_CVG_THRESHOLDS \
    --cw_factor $CW_FACTOR \
    --cw_schedule $CW_SCHEDULE \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_cvg_weighting $USE_CVG_WEIGHTING \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weights $USE_WEIGHTS \
    --weights $WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR \
    --weights_schedule $WEIGHTS_SCHEDULE

