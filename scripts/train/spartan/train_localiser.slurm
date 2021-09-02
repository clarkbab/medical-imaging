#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 2-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASET="PMCC-HN-TRAIN-FLIP-LOC"
MODEL_NAME="localiser-pl"
NUM_GPUS=1
NUM_NODES=1
NUM_WORKERS=4
NUM_SUBSET=None
REGIONS="Parotid_L"
RESUME=False
RESUME_CHECKPOINT=None
RUN_NAME="flipped-baseline"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser --slurm_job_id $SLURM_JOB_ID --model_name $MODEL_NAME --run_name $RUN_NAME --dataset $DATASET --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_subset $NUM_SUBSET --num_workers $NUM_WORKERS --regions $REGIONS --resume $RESUME --resume_checkpoint $RESUME_CHECKPOINT --use_logger $USE_LOGGER
