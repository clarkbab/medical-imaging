#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --partition gpu-a100-short
##SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R','Cochlea_L','Cochlea_R','Lens_L','Lens_R']"   # 0
    "['Brain','OralCavity','BrainStem','Mandible','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"         # 1
    "['SpinalCord','Mandible','BrainStem','OralCavity','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"                        # 4
)
SHORT_REGIONS=(
    'BL_BR_OL_OR_CL_CR_LL_LR'
    'B_OC_BS_M_PL_PR_SL_SR'
    'SC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v3"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"
if [[ $SHORT_REGION == BL_BR_OL_OR_CL_CR_LL_LR ]]; then
    RANDOM_SEED=54
elif [[ $SHORT_REGION == SC ]]; then
    RANDOM_SEED=43
else
    RANDOM_SEED=42
fi
RUN_NAME="8-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
