REGIONS="['Bone_Mandible','Brainstem','Glnd_Submand_L','Glnd_Submand_R','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R']"

# Other params.
MODEL_NAME="sm-numbers"
ARCH="mednext:m"
RESOLUTION="222"
DATASET="MICCAI-$RESOLUTION"
RANDOM_SEED=42
RUN_NAME="$ARCH-$RESOLUTION-regions:ALL-seed:$RANDOM_SEED-lr-1e-3-no-metrics"
SAVE_TRAINING_METRICS=False
LR_INIT=1e-3
N_EPOCHS=500
N_GPUS=1
N_NODES=1
N_WORKERS=8
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

command="python $SCRIPT_DIR/train/segmenter/train_segmenter.py \
    --dataset $DATASET \
    --regions $REGIONS \
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
