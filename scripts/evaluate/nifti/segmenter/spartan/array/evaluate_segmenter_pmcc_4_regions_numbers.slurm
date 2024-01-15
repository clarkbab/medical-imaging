#!/bin/bash
#SBATCH --account punim1413
##SBATCH --partition cascade
#SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R']"     # 0
    "['Brain','OralCavity','BrainStem','Mandible']"                             # 1
    "['Cochlea_L','Cochlea_R','Lens_L','Lens_R']"                               # 2
    "['Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"             # 3
    "['SpinalCord','BrainStem','Parotid_L','Parotid_R']"                        # 4
)
SHORT_REGIONS=(
    'BL_BR_OL_OR'
    'B_OC_BS_M'  
    'CL_CR_LL_LR'
    'PL_PR_SL_SR'
    'SC'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v3"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"
RANDOM_SEED=50
#RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
#RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-ivw-1-schedule-1000"
#RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-ivw-0.5-schedule-100"
RUN_NAME="4-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-lr-1e-4"
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
