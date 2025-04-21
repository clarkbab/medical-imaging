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
MODEL_NAME="sm-convergence"
ARCH="unet3d:m"
RESOLUTION="222"
DATASET="MICCAI-$RESOLUTION"
RANDOM_SEED=42
LR_INIT=1e-4
MODEL_TYPE="SegmenterTversky"
RUN_NAME="$ARCH-$RESOLUTION-regions:1-$SHORT_REGION-seed:$RANDOM_SEED-lr-$LR_INIT-tversky-learn"
SAVE_TRAINING_METRICS=True
N_EPOCHS=1000
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

command="python $SCRIPT_DIR/train/segmenter/train_segmenter.py \
    --dataset $DATASET \
    --regions $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --arch $ARCH \
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
