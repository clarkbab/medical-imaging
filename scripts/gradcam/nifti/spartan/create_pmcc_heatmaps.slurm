#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Choose model.
MODEL="('segmenter-pmcc-numbers-v3','1-region-B-112-seed-42','BEST')"

# Define regions.
MODEL_REGION="('Brain')"
REGION="Brain"

# Define layers.
LAYER="['5','12','19','26','33']"
LAYER_SPACING="[(1,1,2),(2,2,4),(4,4,8),(8,8,16),(16,16,32)]"

# Other params.
DATASET="('PMCC-HN-TEST-BC-112','PMCC-HN-TRAIN-BC-112')"
MODEL_SPACING="(1,1,2)"
CHECK_EPOCHS=True
N_EPOCHS=1000
N_FOLDS=5
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True

python $SCRIPT_DIR/gradcam/nifti/create_heatmaps.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --model_spacing $MODEL_SPACING \
    --region $REGION \
    --layer $LAYER \
    --layer_spacing $LAYER_SPACING \
    --check_epochs $CHECK_EPOCHS \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
