#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:10:00
#SBATCH -A punim1413

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
#PRECISION=32
#RANDOM_SEED=42
RUN_NAME="region-$SHORT_REGION-precision-$PRECISION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
