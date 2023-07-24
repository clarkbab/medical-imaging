#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
## Timing guidelines:
## 1-region-.*-444 model requires ~10 hours for 10k epochs. 12 hours with buffer.
## 1-region-.*-222 model requires ~44 hours for 10k epochs. 2 days with buffer. 
## 1-region-.*-112 model requires ~180 hours for 10k epochs. 8 days with buffer. 
##SBATCH --time 1-06:00:00
#SBATCH --time 0-00:05:00
##SBATCH --time 3-00:00:00
##SBATCH --time 7-00:00:00
##SBATCH --test-only

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
)
SHORT_REGIONS=(
    "ALL"
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-miccai-convergence-part-3"
RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
#RANDOM_SEED=44
#RUN_NAME="9-regions-$SHORT_REGION-scheme-$WEIGHTING_SCHEME-seed-$RANDOM_SEED"
BATCH_SIZE=1
CKPT_MODEL=True
#LR_INIT=1e-3
#N_EPOCHS=10000
N_GPUS=1
N_NODES=1
N_SPLIT_CHANNELS=2
N_WORKERS=8
RESUME=False
RESUME_RUN=None
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_DYNAMIC_WEIGHTS=True
#DYNAMIC_WEIGHTS_FACTOR=0.1
#DYNAMIC_WEIGHTS_CONVERGENCE_DELAY=100
#DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS="[]",
# Remove quotations from string passed via python script.
DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS=$( echo $DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS | tr -d '"' )
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True
USE_LR_SCHEDULER=False
USE_WEIGHTING=False
#WEIGHTING_SCHEME='4c'

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --dynamic_weights_factor $DYNAMIC_WEIGHTS_FACTOR \
    --dynamic_weights_convergence_delay $DYNAMIC_WEIGHTS_CONVERGENCE_DELAY \
    --dynamic_weights_convergence_thresholds $DYNAMIC_WEIGHTS_CONVERGENCE_THRESHOLDS \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_split_channels $N_SPLIT_CHANNELS \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_run $RESUME_RUN \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_dynamic_weights $USE_DYNAMIC_WEIGHTS \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_weighting $USE_WEIGHTING \
    --weighting_scheme $WEIGHTING_SCHEME
