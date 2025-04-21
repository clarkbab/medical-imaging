REGIONS=(
    "['Bone_Mandible','Brainstem']"         # 0
    "['Glnd_Submand_L','Glnd_Submand_R']"   # 1
    "['OpticChiasm','OpticNrv_L']"          # 2
    "['OpticNrv_L','OpticNrv_R']"           # 3
    "['Parotid_L','Parotid_R']"             # 4
)
SHORT_REGIONS=(
    'BM_BS'
    'SL_SR'
    'OC'
    'OL_OR'
    'PL_PR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}

# Other params.
MODEL_NAME="sm-numbers"
ARCH="unet3d:m"
RESOLUTION="222"
DATASET="MICCAI-$RESOLUTION"
RANDOM_SEED=42
RUN_NAME="$ARCH-$RESOLUTION-regions:1-$SHORT_REGION-seed:$RANDOM_SEED-lr-1e-4-no-metrics"
SAVE_TRAINING_METRICS=False
LR_INIT=1e-4
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
