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
    "['BrainStem','OralCavity']"                            # 0
    "['OralCavity','Mandible']"                             # 1

    "['Mandible','SpinalCord']"                             # 2
    "['OralCavity','SpinalCord']"                           # 3
    "['SpinalCord','Submandibular_L']"                      # 4
    "['SpinalCord','Submandibular_R']"                      # 5

    "['BrachialPlexus_L','Cochlea_L']"                      # 6
    "['BrachialPlexus_L','Cochlea_R']"                      # 7
    "['BrachialPlexus_L','Lens_L']"                         # 8
    "['BrachialPlexus_L','Lens_R']"                         # 9
    "['BrachialPlexus_L','OpticNerve_L']"                   # 10
    "['BrachialPlexus_L','OpticNerve_R']"                   # 11
    "['BrachialPlexus_L','Cochlea_L','Cochlea_R']"          # 12
    "['BrachialPlexus_L','Lens_L','Lens_R']"                # 13
    "['BrachialPlexus_L','OpticNerve_L','OpticNerve_R']"    # 14

    "['BrachialPlexus_R','Cochlea_L']"                      # 15
    "['BrachialPlexus_R','Cochlea_R']"                      # 16
    "['BrachialPlexus_R','Lens_L']"                         # 17
    "['BrachialPlexus_R','Lens_R']"                         # 18
    "['BrachialPlexus_R','OpticNerve_L']"                   # 19
    "['BrachialPlexus_R','OpticNerve_R']"                   # 20
    "['BrachialPlexus_R','Cochlea_L','Cochlea_R']"          # 21
    "['BrachialPlexus_R','Lens_L','Lens_R']"                # 22
    "['BrachialPlexus_R','OpticNerve_L','OpticNerve_R']"    # 23
)
SHORT_REGIONS=(
    'BS_OC'
    'OC_M'  

    'M_SC'  
    'OC_S'  
    'SC_SL'  
    'SC_SR'  

    'BL_CL'
    'BL_CR'
    'BL_LL'
    'BL_LR'
    'BL_OL'
    'BL_OR'
    'BL_CL_CR'
    'BL_LL_LR'
    'BL_OL_OR'

    'BR_CL'
    'BR_CR'
    'BR_LL'
    'BR_LR'
    'BR_OL'
    'BR_OR'
    'BR_CL_CR'
    'BR_LL_LR'
    'BR_OL_OR'
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
RANDOM_SEED=43
RUN_NAME_BASE="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME_BASE-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
elif [ "$SHORT_REGION" = "LL_LR" ]; then
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
