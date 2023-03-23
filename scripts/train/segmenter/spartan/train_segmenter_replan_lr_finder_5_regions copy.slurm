#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G 
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-02:00:00

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-112-NATIVE"
REGION="('Bone_Mandible','BrachialPlex_L','Brain','Lens_L','Parotid_L')"
MODEL_NAME="segmenter-replan-112-NATIVE"
TEST_FOLD=0
BATCH_SIZE=1
RUN_NAME="lr-finder-batch-$BATCH_SIZE-$TEST_FOLD"
LR_FIND=True
LR_FIND_MIN_LR=1e-8
LR_FIND_MAX_LR=1e2
LR_FIND_NUM_TRAIN=1000
N_EPOCHS=300
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=False

python $SCRIPT_DIR/train/segmenter/train_segmenter_replan.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --lr_find $LR_FIND \
    --lr_find_min_lr $LR_FIND_MIN_LR \
    --lr_find_max_lr $LR_FIND_MAX_LR \
    --lr_find_num_train $LR_FIND_NUM_TRAIN \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
