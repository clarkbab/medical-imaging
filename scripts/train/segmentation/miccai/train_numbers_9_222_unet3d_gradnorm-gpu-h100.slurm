#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition gpu-h100
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 2-00:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS="['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"

# Other params.
MODEL_NAME="sm-gradnorm"
ARCH="unet3d:m"
RESOLUTION="222"
DATASET="MICCAI-$RESOLUTION"
RANDOM_SEED=42
LOSS_FN="tversky"
LR_INIT=1e-4
MODEL_TYPE="SegmenterGradNorm"
GN_ENABLED=True
GN_ALPHA=1.0
GN_LR_INIT=5e-3
GN_NORM_LIMIT_MULT=None
GN_BALANCE_POINT_TYPE="params"
GN_CLIP_MULT=1.2
RESUME=False
N_EPOCHS=2000
RUN_NAME="$ARCH-$RESOLUTION-regions:ALL-seed:$RANDOM_SEED-lr-$LR_INIT-$LOSS_FN-gradnorm-alpha:$GN_ALPHA-minclips:$GN_CLIP_MULT"
#RUN_NAME="$ARCH-$RESOLUTION-regions:ALL-seed:$RANDOM_SEED-lr-$LR_INIT-$LOSS_FN-gradnorm-alpha:$GN_ALPHA-noclip"
#RUN_NAME="$ARCH-$RESOLUTION-regions:ALL-seed:$RANDOM_SEED-lr-$LR_INIT-$LOSS_FN-gradnorm-disabled"
SAVE_TRAINING_METRICS=False
GN_LOSS_FN="abs"
GN_SOFTMAX=True
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

command="python $SCRIPT_DIR/train/segmentation/train_segmenter.py \
    --dataset $DATASET \
    --regions $REGIONS \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --arch $ARCH \
    --gn_alpha $GN_ALPHA \
    --gn_balance_point_type $GN_BALANCE_POINT_TYPE \
    --gn_clip_mult $GN_CLIP_MULT \
    --gn_enabled $GN_ENABLED \
    --gn_loss_fn $GN_LOSS_FN \
    --gn_lr_init $GN_LR_INIT \
    --gn_norm_limit_mult $GN_NORM_LIMIT_MULT \
    --gn_softmax $GN_SOFTMAX \
    --loss_fn $LOSS_FN \
    --lr_init $LR_INIT \
    --model_type $MODEL_TYPE \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --save_training_metrics $SAVE_TRAINING_METRICS \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID"
echo $command
$command
