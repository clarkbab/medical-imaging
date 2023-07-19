#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
#SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-00:30:00
##SBATCH --test-only

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    'Bone_Mandible'     # 0
    'Brainstem'         # 1
    'Glnd_Submand_L'    # 2
    'OpticChiasm'       # 3
)
SHORT_REGIONS=(
    'BM'
    'BS'
    'SL'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-precision"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
CHECK_EPOCHS=5000
#PRECISION=32
#RANDOM_SEED=42
RUN_NAME="region-$SHORT_REGION-precision-$PRECISION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/predict/nrrd/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
