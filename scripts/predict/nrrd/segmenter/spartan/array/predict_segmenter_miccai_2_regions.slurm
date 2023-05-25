#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
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
RESOLUTION="222"
DATASET="MICCAI-2015-$RESOLUTION"
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
N_SPLIT_CHANNELS=2
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/predict/nrrd/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --n_split_channels $N_SPLIT_CHANNELS \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
