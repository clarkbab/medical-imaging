#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 0
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 3-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

bash $SCRIPT_DIR/train/train_localiser $SLURM_JOB_ID
