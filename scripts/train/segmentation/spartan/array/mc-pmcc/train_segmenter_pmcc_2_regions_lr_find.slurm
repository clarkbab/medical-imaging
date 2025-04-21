#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 1
#SBATCH --time 0-06:00:00
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
    'SC'
    'SL_SR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-lr-find"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-$RESOLUTION','PMCC-HN-TRAIN-$RESOLUTION')"
N_SPLIT_CHANNELS=2
#RANDOM_SEED=42
RUN_NAME="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
BATCH_SIZE=1
CKPT_MODEL=True
LR_FIND=True
LR_FIND_ITER=500
LR_INIT=1e-3
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
N_FOLDS=5
TEST_FOLD=0
USE_LOGGER=False

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --lr_find $LR_FIND \
    --lr_find_iter $LR_FIND_ITER \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER
