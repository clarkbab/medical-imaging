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

DATASET="('PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC')"
REGIONS=(
    "['BrachialPlexus_L','BrachialPlexus_R']"   # 0
    "['Brain','OralCavity']"                    # 1
    "['BrainStem','Mandible']"                  # 2
    "['Cochlea_L','Cochlea_R']"                 # 3
    "['Lens_L','Lens_R']"                       # 4
    "['OpticNerve_L','OpticNerve_R']"           # 5
    "['Parotid_L','Parotid_R']"                 # 6
    "['SpinalCord','Mandible']"                 # 7
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
    'SC_M'
    'SL_SR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers"
RESOLUTION="112"
RANDOM_SEED=42
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
#RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED-lr-1e-4"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
N_FOLDS=5
TEST_FOLD=0

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
