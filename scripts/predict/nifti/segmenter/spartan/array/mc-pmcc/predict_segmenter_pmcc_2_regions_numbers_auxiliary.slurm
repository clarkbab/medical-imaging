#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
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
    "['BrainStem','OralCavity']"                    # 0
    "['OralCavity','Mandible']"                     # 1
)
SHORT_REGIONS=(
    'BS_OC'
    'OC_M'  
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v3"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"
RANDOM_SEED=42
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
CHECK_EPOCHS=True
CROP_MM="(300,400,500)"
N_EPOCHS=1000
N_FOLDS=5
N_SPLIT_CHANNELS=2
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_split_channels $N_SPLIT_CHANNELS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
