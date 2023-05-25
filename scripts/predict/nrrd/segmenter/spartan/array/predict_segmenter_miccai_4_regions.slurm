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
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R']"
    "['OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
    "['OpticChiasm','Bone_Mandible','Brainstem','Glnd_Submand_L']"
)
SHORT_REGIONS=(
    'BM_BS_SL_SR'
    'OL_OR_PL_PR'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai"
RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION"
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
