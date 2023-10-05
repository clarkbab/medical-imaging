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
    'Bone_Mandible'     # 0
    'Brainstem'         # 1
    'Glnd_Submand_L'    # 2
    'Glnd_Submand_R'    # 3
    'OpticChiasm'       # 4
    'OpticNrv_L'        # 5
    'OpticNrv_R'        # 6
    'Parotid_L'         # 7
    'Parotid_R'         # 8
)
SHORT_REGIONS=(
    'BM'
    'BS'
    'SL'
    'SR'
    'OC'
    'OL'
    'OR'
    'PL'
    'PR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-numbers"
RESOLUTION="112"
DATASET="MICCAI-2015-$RESOLUTION"
RANDOM_SEED=42
RUN_NAME="1-region-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
