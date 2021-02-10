#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:4
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

SCRIPT_DIR=/home/baclark/code/medical-imaging/scripts

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

python $SCRIPT_DIR/parotid_left_2d_train_node --print-interval 200 --record-interval 200 --validation-interval epoch \
    --num-nodes 1 --use-gpu --num-gpus 1
