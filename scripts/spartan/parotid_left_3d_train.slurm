#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:4
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

PROJECT_DIR="/data/projects/punim1413/"
SCRIPT_DIR="$PROJECT_DIR/medical-imaging/scripts"

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

python $SCRIPT_DIR/parotid_left_3d_train_node --print-interval 10 --record-interval 50 --validation-interval epoch \
    --num-nodes 1 --use-gpu --num-gpus 1 --use-mixed --max-epochs 300
