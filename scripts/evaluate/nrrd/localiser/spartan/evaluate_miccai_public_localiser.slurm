#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 120G
##SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="MICCAI-2015"
REGION="Brain"
LOCALISER="('localiser-$REGION','public-1gpu-150epochs','best')"
METRICS="['pred-extent-mm']"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nrrd/localiser/create_evaluation.py \
    --dataset $DATASET \
    --region $REGION \
    --localiser $LOCALISER \
    --metrics $METRICS
