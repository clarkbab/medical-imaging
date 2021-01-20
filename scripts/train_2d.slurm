#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:4
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

SCRIPT_DIR=/home/baclark/code/medical-imaging/scripts
PROJECT_DIR=/data/gpfs/projects/punim1413
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints
TENSORBOARD_DIR=$PROJECT_DIR/tensorboard
TRAIN_DATA_DIR=$PROJECT_DIR/HEAD-NECK-RADIOMICS-HN1/processed/2d-parotid-left

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

python $SCRIPT_DIR/train_2d --print-interval 200 --record-interval 200 --validation-interval epoch \
    --checkpoint-dir $CHECKPOINT_DIR --tensorboard_dir $TENSORBOARD_DIR --data-dir $TRAIN_DATA_DIR \
    --num-nodes 1 --num-gpus 4
