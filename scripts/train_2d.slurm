#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH -A punim1413
#SBATCH -q gpgpumse

PROJECT_DIR=/data/gpfs/projects/punim1413
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints
TENSORBOARD_DIR=$PROJECT_DIR/tensorboard
TRAIN_DATA_DIR=$PROJECT_DIR/HEAD-NECK-RADIOMICS-HN1/processed/2d-parotid-left

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

# Set current directory to find 'train_2d' script.
current_dir=$(dirname "$(realpath "$0")")""

CHECKPOINT_DIR=$CHECKPOINT_DIR TRAIN_DATA=$TRAIN_DATA_DIR TENSORBOARD_DIR=$TENSORBOARD_DIR python $current_dir/train_2d --print-interval 200 --record-interval 200 --validation-interval epoch

