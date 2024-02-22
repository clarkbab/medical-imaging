#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 120G
##SBATCH --gres gpu:1
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set regions.
REGIONS=(
    "['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R','Cochlea_L','Cochlea_R','Lens_L','Lens_R']"   # 0
    "['Brain','OralCavity','BrainStem','Mandible','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"         # 1
    "['SpinalCord','Mandible','BrainStem','OralCavity','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"    # 2
)
SHORT_REGIONS=(
    'BL_BR_OL_OR_CL_CR_LL_LR'
    'B_OC_BS_M_PL_PR_SL_SR'
    'SC'
)
MODEL_REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
#TARGET_REGION=<set by --export=ALL,TARGET_REGION=...>

# Dataset.
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"

# Define model.
MODEL_NAME="segmenter-pmcc-numbers-v3"
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    RANDOM_SEED=54
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    RANDOM_SEED=43
else
    RANDOM_SEED=42
fi
RUN_NAME="8-regions-$MODEL_SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
MODEL_SPACING="(1,1,2)"

# Set loader params.
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0

# Define layers.
LAYER="['5','12','19','26','33','40','47','54','61']"
#LAYER="['5','12','19','26','33']"
#LAYER="['33']"

# Other params.
AUX_REGION="RL:PMCC"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nifti/gradcam/create_multi_segmenter_heatmap_evaluation.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --target_region $TARGET_REGION \
    --layer $LAYER \
    --aux_region $AUX_REGION \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD
