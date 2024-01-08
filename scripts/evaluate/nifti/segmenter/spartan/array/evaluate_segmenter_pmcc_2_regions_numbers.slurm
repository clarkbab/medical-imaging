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
    "['BrachialPlexus_L','BrachialPlexus_R']"   # 0
    "['Brain','OralCavity']"                    # 1
    "['BrainStem','Mandible']"                  # 2
    "['Cochlea_L','Cochlea_R']"                 # 3
    "['Lens_L','Lens_R']"                       # 4
    "['OpticNerve_L','OpticNerve_R']"           # 5
    "['Parotid_L','Parotid_R']"                 # 6
    "['SpinalCord','BrainStem']"                # 7
    "['Submandibular_L','Submandibular_R']"     # 8
)
SHORT_REGIONS=(
    'BL_BR'
    'B_OC'  
    'BS_M'
    'CL_CR'
    'LL_LR'
    'OL_OR'
    'PL_PR'
    'SC'
    'SL_SR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v2"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-$RESOLUTION','PMCC-HN-TRAIN-$RESOLUTION')"
RANDOM_SEED=48
#RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-lr-1e-4"
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
