#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition snowy
#SBATCH --time 0-2:00:00
#SBATCH --mem 16G
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/quick/pmcc_hn_train_seg_generate_loc
