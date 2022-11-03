#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-01:00:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
DATASETS="['PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC']"
N_PATS=50

python $SCRIPT_DIR/process/nifti/combine_segmenter_predictions.py \
    --dataset $DATASETS \
    --n_pats $N_PATS \
