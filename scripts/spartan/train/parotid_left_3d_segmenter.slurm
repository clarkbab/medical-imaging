#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:p100:4
#SBATCH --time 7-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

RUN_NAME="with_affine_aug"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/train/parotid_left_3d_segmenter --name $RUN_NAME --train-print-interval 50 --train-report-interval 50
