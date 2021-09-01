#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:4
#SBATCH --cpus-per-gpu 4
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

REGIONS=('Brain' 'Cochlea_L' 'Cochlea_R' 'MedullaOblongata' 'Neck_L' 'Neck_R' 'OralCavity' 'Parotid_L' 'Parotid_R' 'SpinalCord' 'Submandibular_L' 'Submandibular_R')
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
DATASET="HN1-SEG"
MODEL_NAME="segmenter-${REGION}"
NUM_GPUS=4
NUM_NODES=1
NUM_WORKERS=4
RUN_NAME="hn1-200epoch"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter --slurm_job_id $SLURM_JOB_ID --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name $MODEL_NAME --run_name $RUN_NAME --dataset $DATASET --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_workers $NUM_WORKERS --regions $REGIONS --use_logger $USE_LOGGER
