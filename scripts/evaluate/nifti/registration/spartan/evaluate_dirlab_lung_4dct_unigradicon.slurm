#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="DIRLAB-LUNG-4DCT"
MODEL="unigradicon"
LANDMARKS="all"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

command="python $SCRIPT_DIR/evaluate/nifti/registration/evaluate_registrations.py \
    --dataset $DATASET \
    --model $MODEL \
    --landmarks $LANDMARKS"
echo $command
$command
