#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 256G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --test-only

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-222"
REGION="all"
MODEL_NAME="segmenter-replan-222"
TEST_FOLD=0
HALVE_CHANNELS=True
RUN_NAME="allregions-weightdecay-$TEST_FOLD"
BATCH_SIZE=1
CKPT_MODEL=True
LR_INIT=1e-3
LR_SCHEDULER=True
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True
WEIGHT_DECAY=1e-3

python $SCRIPT_DIR/train/segmenter/train_segmenter_replan.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --halve_channels $HALVE_CHANNELS \
    --lr_init $LR_INIT \
    --lr_scheduler $LR_SCHEDULER \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER \
    --weight_decay $WEIGHT_DECAY
