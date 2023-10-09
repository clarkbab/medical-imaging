#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['Bone_Mandible','Brainstem','Parotid_L','Parotid_R']"   # 0
    "['Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R']"               # 1
    "['OpticChiasm','OpticNrv_L','OpticNrv_R','Glnd_Submand_L']"        # 2
)
SHORT_REGIONS=(
    'BM_BS_PL_PR'
    'SL_SR_OL_OR'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-numbers-boost"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
N_SPLIT_CHANNELS=2
RANDOM_SEED=42
LR_INIT=1e-4
RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-lr-$LR_INIT"
BATCH_SIZE=1
CKPT_MODEL=True
N_EPOCHS=7000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=True
RESUME_CKPT='last'
RESUME_MODEL="segmenter-miccai-numbers"
RESUME_RUN="1-region-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
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
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_model $RESUME_MODEL \
    --resume_run $RESUME_RUN \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER
