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
    "['BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R']"           # 0
)
SHORT_REGIONS=(
    'ALL'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-resolution-v2"
RESOLUTION="444"
DATASET="('PMCC-HN-TEST-$RESOLUTION','PMCC-HN-TRAIN-$RESOLUTION')"
RANDOM_SEED=42
RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-ivw-0.5-schedule-200"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
CHECK_EPOCHS=False
N_EPOCHS=1000
N_FOLDS=5
N_SPLIT_CHANNELS=2
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True

python $SCRIPT_DIR/evaluate/nifti/segmenter/evaluate_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
