#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpu-a100
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 0-0:30:00

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-TRAIN-REPLAN-DOUBLE-LOC"
REGION="Brain"
MODEL_NAME="replan-localiser-${REGION}"
n_EPOCHS=300
n_GPUS=1
n_NODES=1
n_WORKERS=4
RESUME=False
RESUME_CHECKPOINT='last'
RUN_NAME="test"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TRUNCATE_SPINE=True
USE_LOGGER=False

python $SCRIPT_DIR/train/localiser/train_replan_localiser.py \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --dataset $DATASET \
    --n_epochs $n_EPOCHS \
    --n_gpus $n_GPUS \
    --n_nodes $n_NODES \
    --n_workers $n_WORKERS \
    --region $REGION \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --truncate_spine $TRUNCATE_SPINE \
    --use_logger $USE_LOGGER
