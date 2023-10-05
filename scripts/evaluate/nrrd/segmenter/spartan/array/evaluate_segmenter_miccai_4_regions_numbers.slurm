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
    "['Bone_Mandible','Brainstem','Parotid_L','Parotid_R']"   # 0
    "['Glnd_Submand_L','Glnd_Submand_R','OpticNrv_L','OpticNrv_R']"               # 1
    "['OpticChiasm','OpticNrv_L','OpticNrv_R','Glnd_Submand_L']"        # 2
)
SHORT_REGIONS=(
    'BM_BS_PL_PR'
    'SL_SR_OL_OR'
    'OC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-numbers"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
RANDOM_SEED=42
RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
