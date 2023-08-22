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
## Timing guidelines:
## 1-region-.*-444 model requires ~10 hours for 10k epochs. 12 hours with buffer.
## 1-region-.*-222 model requires ~44 hours for 10k epochs. 2 days with buffer. 
## 1-region-.*-112 model requires ~180 hours for 10k epochs. 8 days with buffer. 
##SBATCH --time 1-06:00:00
##SBATCH --time 4-00:00:00
##SBATCH --time 7-00:00:00
#SBATCH --time 0-01:00:00
##SBATCH --test-only

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['BrachialPlexus_L','BrachialPlexus_R']"   # 0
    "['Brain','BrainStem']"                     # 1
    "['Cochlea_L','Cochlea_R']"                 # 2
    "['Lens_L','Lens_R']"                       # 3
    "['Mandible','OralCavity']"                 # 4
    "['OpticNerve_L','OpticNerve_R']"           # 5
    "['Parotid_L','Parotid_R']                  # 6
    "['SpinalCord','BrachialPlexus_L']          # 7
    "['Submandibular_L','Submandibular_R']"     # 8
)
SHORT_REGIONS=(
    'BL_BR'
    'B_BS'
    'CL_CR'
    'LL_LR'
    'M_OC'
    'OL_OR'
    'PL_PR'
    'SC'
    'SL_SR'
)
DW_CVG_THRESHOLDSES=(
    "[0.4,0.4]"
    "[0.9,0.8]"
    "[0.4,0.4]"
    "[0.6,0.6]"
    "[0.8,0.7]"
    "[0.6,0.6]"
    "[0.7,0.7]"
    "[0.7,0.4]"
    "[0.7,0.7]"
)
DW_CVG_THRESHOLDS=${DW_CVG_THRESHOLDSES[$SLURM_ARRAY_TASK_ID]}
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers"
RESOLUTION="112"
DATASET="['PMCC-HN-TEST-$RESOLUTION','PMCC-HN-TRAIN-$RESOLUTION']"
N_SPLIT_CHANNELS=2
RANDOM_SEED=42
RUN_NAME="2-regions-$SHORT_REGION-seed-$RANDOM_SEED"
BATCH_SIZE=1
CKPT_MODEL=True
LR_INIT=1e-3
N_EPOCHS=150
N_FOLDS=5
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_DOWNWEIGHTING=True
DW_FACTOR=100
DW_CVG_DELAY_ABOVE=20
DW_CVG_DELAY_BELOW=5
USE_LOADER_SPLIT_FILE=False
USE_LOGGER=True
USE_LR_SCHEDULER=False
USE_WEIGHTS=False
WEIGHTS_IV_FACTOR=2

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --dw_cvg_delay_above $DW_CVG_DELAY_ABOVE \
    --dw_cvg_delay_below $DW_CVG_DELAY_BELOW \
    --dw_cvg_thresholds $DW_CVG_THRESHOLDS \
    --dw_factor $DW_FACTOR \
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
    --use_downweighting $USE_DOWNWEIGHTING \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weights $USE_WEIGHTS \
    --weights $WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR
