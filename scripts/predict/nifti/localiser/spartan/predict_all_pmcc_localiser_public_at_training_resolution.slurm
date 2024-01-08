#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="('PMCC-HN-TEST','PMCC-HN-TRAIN')"
REGION="Brain"
LOCALISER="('localiser-$REGION','public-1gpu-150epochs','BEST')"
N_EPOCHS=150
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TIMING=True

python $SCRIPT_DIR/predict/nifti/localiser/create_all_predictions_at_training_resolution.py \
    --dataset $DATASET \
    --localiser $LOCALISER \
    --n_epochs $N_EPOCHS \
    --timing $TIMING 