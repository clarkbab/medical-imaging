#!/bin/bash
#SBATCH --nodes 1
#SBATCH --patition gpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=1
#SBATCH -A punim1413
#SBATCH -q gpgpumse

export TRAIN_DATA=/data/gpfs/projects/punim1413/HEAD-NECK-RADIOMICS-HN1/processed/2d-parotid-left

module load gcccore/8.3.0
module load python/3.8.2

python --version

