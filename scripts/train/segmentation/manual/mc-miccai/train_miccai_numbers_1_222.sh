#!/bin/bash

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    'Bone_Mandible'     # 0
    'Brainstem'         # 1
    'Glnd_Submand_L'    # 2
    'Glnd_Submand_R'    # 3
    'OpticChiasm'       # 4
    'OpticNrv_L'        # 5
    'OpticNrv_R'        # 6
    'Parotid_L'         # 7
    'Parotid_R'         # 8
)
SHORT_REGIONS=(
    'BM'
    'BS'
    'SL'
    'SR'
    'OC'
    'OL'
    'OR'
    'PL'
    'PR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}

# Other params.
MODEL_NAME="sm-numbers"
RESOLUTION="222"
DATASET="MICCAI-$RESOLUTION"
RANDOM_SEED=42
RUN_NAME="$RESOLUTION-regions:1-$SHORT_REGION-seed:$RANDOM_SEED"
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
PRECISION=32
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_WANDB=True

python $SCRIPT_DIR/train/segmenter/train_mednext.py \
    --dataset $DATASET \
    --regions $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --precision $PRECISION \
    --random_seed $RANDOM_SEED \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --slurm_job_id $SLURM_JOB_ID \
    --use_wandb $USE_WANDB
