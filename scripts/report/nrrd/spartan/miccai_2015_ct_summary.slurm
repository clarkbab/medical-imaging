#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --mem 32G
#SBATCH --time 0-00:30:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="MICCAI-2015"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nrrd/create_ct_summary.py \
    --dataset $DATASET
