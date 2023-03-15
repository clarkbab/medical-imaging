#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:10:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-LOC"
REGION="Brain"
N_FOLDS=5
TEST_FOLD=0
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/loaders/create_multi_loader_manifest.py \
    --dataset $DATASET \
    --region $REGION \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD
