#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:4
#SBATCH --time 2-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

RUN_NAME="with_hausdorff"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/train/parotid_left_3d --name $RUN_NAME --print-interval 10 --record-interval 50 \
    --validation-interval epoch --num-nodes 1 --num-gpus 1
