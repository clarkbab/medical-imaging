#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
#SBATCH --gres gpu:p100:4
#SBATCH --cpus-per-gpu 4
#SBATCH --time 2-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['HN1-LOC','HNPCT-LOC','HNSCC-LOC','OPC-LOC']"
REGIONS=(
    'BrachialPlexus_L'
    'BrachialPlexus_R'
    'Brain'
    'BrainStem'
    'Cochlea_L'
    'Cochlea_R'
    'Lens_L'
    'Lens_R'
    'Mandible'
    'OpticNerve_L'
    'OpticNerve_R'
    'OralCavity'
    'Parotid_L'
    'Parotid_R'
    'SpinalCord'
    'Submandibular_L'
    'Submandibular_R'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="localiser-${REGION}"
NUM_EPOCHS=400
NUM_GPUS=4
NUM_NODES=1
NUM_WORKERS=4
RESUME=False
RESUME_CHECKPOINT=None
RUN_NAME="public-large-400epochs"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser --slurm_array_job_id $SLURM_ARRAY_JOB_ID --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --num_epochs $NUM_EPOCHS \
    --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_workers $NUM_WORKERS --region $REGION --resume $RESUME --resume_checkpoint $RESUME_CHECKPOINT --use_logger $USE_LOGGER
