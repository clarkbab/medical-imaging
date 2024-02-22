#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --mem 32G
#SBATCH --time 0-06:00:00
#SBATCH -A punim1413

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

DATASET="MICCAI-2015-BC-112"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/training/create_ct_figures.py \
    --dataset $DATASET
