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
    "['Bone_Mandible','Brainstem']"         # 0
    "['Glnd_Submand_L','Glnd_Submand_R']"   # 1
    "['OpticChiasm','Bone_Mandible']"       # 2
    "['OpticNrv_L','OpticNrv_R']"           # 3
    "['Parotid_L','Parotid_R']"             # 4
)
SHORT_REGIONS=(
    'BM_BS'
    'SL_SR'
    'OC'
    'OL_OR'
    'PL_PR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $REGION \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
