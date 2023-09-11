#!/bin/bash

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
REGIONS=(
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
)
SHORT_REGIONS=(
    "ALL"
)
REGION=${REGIONS[0]}
REGION="['Brainstem','OpticNrv_L']"
SHORT_REGION=${SHORT_REGIONS[0]}
MODEL_NAME="test"
USE_CVG_WEIGHTING=False
CW_FACTOR="[1,0]"
CW_CVG_DELAY_ABOVE=20
CW_CVG_DELAY_BELOW=5
CW_CVG_THRESHOLDS="[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]"
CW_SCHEDULE="[0,10]"
USE_WEIGHTS=False
WEIGHTS_IV_FACTOR="[1,0]"
WEIGHTS_SCHEDULE="[0,10]"
RANDOM_SEED=42
N_SPLIT_CHANNELS=2
RUN_NAME="test-loss-reduced-True-b"
LR_FIND=False
LR_INIT=1e-3
LR_SCHEDULER=False
N_EPOCHS=20
N_GPUS=1
N_NODES=1
N_WORKERS=8
PRECISION='bf16'
BATCH_SIZE=1
RESUME=False
RESUME_RUN=None
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --cw_factor $CW_FACTOR \
    --cw_cvg_delay_above $CW_CVG_DELAY_ABOVE \
    --cw_cvg_delay_below $CW_CVG_DELAY_BELOW \
    --cw_cvg_thresholds $CW_CVG_THRESHOLDS \
    --cw_schedule $CW_SCHEDULE \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_run $RESUME_RUN \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_cvg_weighting $USE_CVG_WEIGHTING \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weights $USE_WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR \
    --weights_schedule $WEIGHTS_SCHEDULE

