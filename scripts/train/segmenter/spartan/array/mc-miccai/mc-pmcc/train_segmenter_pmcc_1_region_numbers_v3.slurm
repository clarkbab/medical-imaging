#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
##SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
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
N_SPLIT_CHANNELS=2
RESUME=False
LR_INIT=1e-3

# Get run name.
RANDOM_SEED=44
RUN_NAME_BASE="1-region-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME_BASE-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
elif [ "$SHORT_REGION" = "LL" ] || [ "$SHORT_REGION" = "LR" ]; then
    LR_INIT=1e-4
    RUN_NAME="$RUN_NAME_BASE-lr-$LR_INIT"
else
    RUN_NAME="$RUN_NAME_BASE"
fi

BATCH_SIZE=1
CKPT_MODEL=True
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLD=0
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --load_all_samples $LOAD_ALL_SAMPLES \
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
