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
#SBATCH --time 0-00:30:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="MICCAI-2015"
PAT_ID="0522c0001"
MODEL="('segmenter-miccai-numbers','1-region-BM-112-seed-42','best')"
MODEL_REGION="Brain"
MODEL_SPACING="(1,1,2)"
REGION="Brain"
LAYER="['5','12','19','26','33']"
LAYER_SPACING="[(1,1,2),(2,2,4),(4,4,8),(8,8,16),(16,16,32)]"
CHECK_EPOCHS=True
N_EPOCHS=5000
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/gradcam/nrrd/create_heatmap.py \
    --dataset $DATASET \
    --pat_id $PAT_ID \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --model_spacing $MODEL_SPACING \
    --region $REGION \
    --layer $LAYER \
    --layer_spacing $LAYER_SPACING \
    --check_epochs $CHECK_EPOCHS \
    --n_epochs $N_EPOCHS

