#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
##SBATCH --mem 128G
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-00:30:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
    "['OpticChiasm','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
)
SHORT_REGIONS=(
    'BM_BS_SL_SR_OL_OR_PL_PR'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-numbers"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
RANDOM_SEED=42
#RUN_NAME="8-regions-BM_BS_SL_SR_OL_OR_PL_PR-112-seed-42-cw-0-ivw-0.5-schedule-1500"
RUN_NAME="8-regions-OC-112-seed-43-cw-2-ivw-0.5-schedule-1000"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
