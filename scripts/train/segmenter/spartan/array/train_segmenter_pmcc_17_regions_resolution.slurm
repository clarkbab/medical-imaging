#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 16
##SBATCH --time 1-00:00:00
##SBATCH --time 3-00:00:00
#SBATCH --time 7-00:00:00
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
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-$RESOLUTION','PMCC-HN-TRAIN-$RESOLUTION')"
N_SPLIT_CHANNELS=2
USE_CVG_WEIGHTING=False
CW_CVG_DELAY_ABOVE=20
CW_CVG_DELAY_BELOW=5
CW_FACTOR_0=2
CW_FACTOR="[$CW_FACTOR_0,0]"
WEIGHTS=None
WEIGHTS_IV_FACTOR_0=0.5
WEIGHTS_IV_FACTOR="[$WEIGHTS_IV_FACTOR_0,0]"
RESUME=False
LR_INIT=1e-3
RANDOM_SEED=42
USE_WEIGHTS=False
SCHEDULE_EPOCHS=200
GRAD_ACC=1
LAM=0.5
USE_LR_SCHEDULER=False
CYCLIC_MIN=1e-4
CYCLIC_MAX=1e-3
USE_ELASTIC=False
RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-acc-$GRAD_ACC"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-lam-$LAM"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-cyclic-$CYCLIC_MIN-$CYCLIC_MAX"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-elastic"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-cw-$CW_FACTOR_0-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
#RESUME_RUN="$RESOLUTION-seed-$RANDOM_SEED-ivw-$WEIGHTS_IV_FACTOR_0-schedule-1000"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS-lr-$LR_INIT"
#RUN_NAME="$RESOLUTION-seed-$RANDOM_SEED-lr-$LR_INIT"
N_EPOCHS=1000
WEIGHTS_SCHEDULE="[0,$SCHEDULE_EPOCHS]"
CW_SCHEDULE="[0,$SCHEDULE_EPOCHS]"
BATCH_SIZE=1
CKPT_MODEL=True
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME_RUN=None
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=False
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
    --cw_cvg_delay_above $CW_CVG_DELAY_ABOVE \
    --cw_cvg_delay_below $CW_CVG_DELAY_BELOW \
    --cw_factor $CW_FACTOR \
    --cw_schedule $CW_SCHEDULE \
    --cyclic_min $CYCLIC_MIN \
    --cyclic_max $CYCLIC_MAX \
    --grad_acc $GRAD_ACC \
    --lam $LAM \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_run $RESUME_RUN \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --test_fold $TEST_FOLD \
    --use_cvg_weighting $USE_CVG_WEIGHTING \
    --use_elastic $USE_ELASTIC \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weights $USE_WEIGHTS \
    --weights $WEIGHTS \
    --weights_iv_factor $WEIGHTS_IV_FACTOR \
    --weights_schedule $WEIGHTS_SCHEDULE
