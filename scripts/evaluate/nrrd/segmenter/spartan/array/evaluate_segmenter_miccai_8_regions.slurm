#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:20:00
#SBATCH -A punim1413

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
    "['OpticChiasm','Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L']"
)
SHORT_REGIONS=(
    'BM_BS_SL_SR_OL_OR_PL_PR'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
RUN_NAME="8-regions-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $REGION \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
