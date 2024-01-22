#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
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

# Set regions.
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
SHORT_REGIONS=(
    'BL'
    'BR'
    'B'  
    'BS'
    'CL'
    'CR'
    'LL'
    'LR'
    'M' 
    'OL'
    'OR'
    'OC' 
    'PL'
    'PR'
    'SC'
    'SL'
    'SR'
)
MODEL_REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
TARGET_REGION=$MODEL_REGION

# Dataset.
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"

# Define model.
MODEL_NAME="segmenter-pmcc-numbers-v3"
RANDOM_SEED=42
RUN_NAME="1-region-$MODEL_SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [[ $SLURM_ARRAY_TASK_ID -eq 6 || $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
    RUN_NAME="$RUN_NAME-lr-1e-4"
fi
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
MODEL_SPACING="(1,1,2)"

# Set loader params.
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False

# Define layers.
LAYER="['5','12','19','26','33','40','47','54','61']"
LAYER_SPACING="[(1,1,2),(2,2,4),(4,4,8),(8,8,16),(16,16,32),(8,8,16),(4,4,8),(2,2,4),(1,1,2)]"
#LAYER="['5','12','19','26','33']"
#LAYER_SPACING="[(1,1,2),(2,2,4),(4,4,8),(8,8,16),(16,16,32)]"
#LAYER="['33']"
#LAYER_SPACING="[(16,16,32)]"

# Other params.
CHECK_EPOCHS=True
CROP_MM="(300,400,500)"
N_EPOCHS=1000
SAVE_TMP_FILES=False
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/gradcam/nifti/create_multi_segmenter_heatmaps.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --model_spacing $MODEL_SPACING \
    --target_region $TARGET_REGION \
    --layer $LAYER \
    --layer_spacing $LAYER_SPACING \
    --check_epochs $CHECK_EPOCHS \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --save_tmp_files $SAVE_TMP_FILES \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
