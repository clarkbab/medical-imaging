#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413

PROJECT_DIR="/data/projects/punim1413/"
SCRIPT_DIR="$PROJECT_DIR/medical-imaging/scripts"

module load gcccore/8.3.0
module load python/3.8.2

version=$(python --version)
echo $version

python $SCRIPT_DIR/preprocess/parotid_left_3d
