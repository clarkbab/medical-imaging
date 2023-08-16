#!/bin/bash

module load python/3.8.6
module load web_proxy
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
SHORT_REGION=${SHORT_REGIONS[0]}
MODEL_NAME="test"
USE_COMPLEXITY_WEIGHTS=False
COMPLEXITY_WEIGHTS_FACTOR=2
COMPLEXITY_WEIGHTS_WINDOW=5
USE_DOWNWEIGHTING=True
DW_FACTOR=10
DW_CVG_DELAY_ABOVE=20
DW_CVG_DELAY_BELOW=5
DW_CVG_THRESHOLDS="[0.8,0.8,0.6,0.6,0.3,0.4,0.4,0.7,0.7]"
RANDOM_SEED=43
# RUN_NAME="factor-$DW_FACTOR-delay-$DYNAMIC_WEIGHTS_CVG_DELAY-seed-$RANDOM_SEED-test"
RUN_NAME="test-resume-part-2"
LR_FIND=False
LR_INIT=1e-3
LR_SCHEDULER=False
N_EPOCHS=4
N_GPUS=1
N_NODES=1
N_SPLIT_CHANNELS=2
N_WORKERS=8
PRECISION='bf16'
BATCH_SIZE=1
RESUME=True
RESUME_RUN="test-resume-part-1"
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True
USE_WEIGHTS=True
WEIGHTS="[0.1,0.1,0.1,0.1,1,1,1,0.1,0.1]"

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --complexity_weights_factor $COMPLEXITY_WEIGHTS_FACTOR \
    --complexity_weights_window $COMPLEXITY_WEIGHTS_WINDOW \
    --dw_factor $DW_FACTOR \
    --dw_cvg_delay_above $DW_CVG_DELAY_ABOVE \
    --dw_cvg_delay_below $DW_CVG_DELAY_BELOW \
    --dw_cvg_thresholds $DW_CVG_THRESHOLDS \
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
    --use_complexity_weights $USE_COMPLEXITY_WEIGHTS \
    --use_downweighting $USE_DOWNWEIGHTING \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weights $USE_WEIGHTS \
    --weights $WEIGHTS

