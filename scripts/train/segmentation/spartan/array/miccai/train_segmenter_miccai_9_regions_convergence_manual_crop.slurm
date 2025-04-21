#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
##SBATCH --time 7-00:00:00
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGION="['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"
MODEL_NAME="segmenter-miccai-convergence"
RESOLUTION="112"
DATASET="MICCAI-2015-MC-$RESOLUTION"
#DATASET="MICCAI-2015-MC-100"
RANDOM_SEED=42
LOSS_FN="dice"
LR_INIT=2e-3
OPTIMISER="rmsprop"
USE_AFFINE_NORM=False
USE_ELASTIC=True
USE_INIT=False
USE_SINGLE_DOWNSAMPLE=False
RUN_NAME="manual-crop-baseline-elastic-seed-$RANDOM_SEED"
BATCH_SIZE=1
CKPT_MODEL=True
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
PRELOAD_DATA=True
PROFILER=None
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True
USE_LOGGER=True
USE_LR_SCHEDULER=False

python $SCRIPT_DIR/train/segmenter/train_multi_segmenter_convergence.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --batch_size $BATCH_SIZE \
    --ckpt_model $CKPT_MODEL \
    --loss_fn $LOSS_FN \
    --lr_init $LR_INIT \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --optimiser $OPTIMISER \
    --preload_data $PRELOAD_DATA \
    --profiler $PROFILER \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_affine_norm $USE_AFFINE_NORM \
    --use_elastic $USE_ELASTIC \
    --use_init $USE_INIT \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE \
    --use_logger $USE_LOGGER \
    --use_lr_scheduler $USE_LR_SCHEDULER \
    --use_single_downsample $USE_SINGLE_DOWNSAMPLE
