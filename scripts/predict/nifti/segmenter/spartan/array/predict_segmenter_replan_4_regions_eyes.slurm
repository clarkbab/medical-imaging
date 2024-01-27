#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Get region/short region.
REGIONS=(
    "['Eye_L','Eye_R','Lens_L','Lens_R']"     # 0
)
SHORT_REGIONS=(
    'EL_ER_LL_LR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}

# Get model name.
RESOLUTION="112"
DATASET="PMCC-HN-REPLAN-EYES-$RESOLUTION"
MODEL_NAME="segmenter-replan-eyes"

# Get run name.
LR_INIT=1e-4
RANDOM_SEED=43
SCHEDULE_EPOCHS=200
USE_WEIGHTS=True
WEIGHTS_IV_FACTOR_0=1.0
RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
fi
if [ "$LR_INIT" = 1e-4 ]; then
    RUN_NAME="$RUN_NAME-lr-$LR_INIT"
fi

# Loader settings.
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False

# Other settings.
CHECK_EPOCHS=True
CROP_MM="(330,380,500)"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
N_EPOCHS=1000
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
