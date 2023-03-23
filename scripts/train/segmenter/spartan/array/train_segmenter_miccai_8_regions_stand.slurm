#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 2-00:00:00
##SBATCH --test-only

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
)
SHORT_REGIONS=(
    "ALL"
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-stand"
RESOLUTION="222"
DATASET="MICCAI-2015-$RESOLUTION"
N_SPLIT_CHANNELS=1
RUN_NAME="baseline-8-regions-$SHORT_REGION-$RESOLUTION"
BATCH_SIZE=1
CKPT_MODEL=True
LR_INIT=1e-3
LR_SCHEDULER=False
N_EPOCHS=2000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True
USE_STAND=True

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --lr_init $LR_INIT \
    --lr_scheduler $LR_SCHEDULER \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_job_id $SLURM_JOB_ID \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_stand $USE_STAND
