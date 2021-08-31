#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpgpu
#SBATCH --mem=0
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-gpu 4
#SBATCH --time=1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASET="PMCC-HN-TRAIN-LOC"
MODEL_NAME="localiser-pl"
NUM_GPUS=2
NUM_NODES=1
NUM_WORKERS=4
REGIONS="Parotid_L"
RUN_NAME="baseline-4gpu"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser --model_name $MODEL_NAME --run_name $RUN_NAME --dataset $DATASET --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_workers $NUM_WORKERS --regions $REGIONS --use_logger $USE_LOGGER
