#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition feit-gpu-a100
#SBATCH --mem 128G 
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 2-00:00:00
#SBATCH -A punim1413
#SBATCH -q feit

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-112-NATIVE"
# REGION="('Bone_Mandible','BrachialPlex_L','Brain','Lens_L','Parotid_L')"
# REGION="('Bone_Mandible','Brain','Lens_L','Parotid_L')"
# REGION="('Bone_Mandible','Brain','Parotid_L')"
# REGION="('Brain','Parotid_L')"
REGION="Brain"
MODEL_NAME="segmenter-replan"
TEST_FOLD=0
HALVE_CHANNELS=True
RUN_NAME="one-region-$TEST_FOLD"
LR_FIND=False
N_EPOCHS=200
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter/train_segmenter_replan.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --halve_channels $HALVE_CHANNELS \
    --lr_find $LR_FIND \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
