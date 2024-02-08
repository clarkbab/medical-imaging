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
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v3"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"
RANDOM_SEED=42
RUN_NAME="1-region-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [[ $SHORT_REGION == LL ]] || [[ $SHORT_REGION == LR ]]; then
    RUN_NAME="$RUN_NAME-lr-1e-4"
fi
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
