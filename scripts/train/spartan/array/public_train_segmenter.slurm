#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 64G
#SBATCH --gres gpu:p100:4
#SBATCH --cpus-per-gpu 4
#SBATCH --time 7-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['HN1-SEG','HNPCT-SEG','HNSCC-SEG','OPC-SEG']"
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
PATCH_SIZES=(
    '(200,115,70)' # 'BrachialPlexus_L'
    '(200,115,70)' # 'BrachialPlexus_R'
    '(170,205,90)' # 'Brain'
    '(95,85,60)' # 'BrainStem'
    '(70,65,25)' # 'Cochlea_L'
    '(70,65,25)' # 'Cochlea_R'
    '(55,50,25)' # 'Lens_L'
    '(55,50,25)' # 'Lens_R'
    '(180,170,75)' # 'Mandible'
    '(70,90,30)' # 'OpticNerve_L'
    '(70,90,30)' # 'OpticNerve_R'
    '(180,160,155)' # 'OralCavity'
    '(95,120,60)' # 'Parotid_L'
    '(95,120,60)' # 'Parotid_R'
    '(80,180,190)' # 'SpinalCord'
    '(75,85,50)' # 'Submandibular_L'
    '(75,85,50)' # 'Submandibular_R'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
PATCH_SIZE=${PATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-${REGION}"
NUM_EPOCHS=400
NUM_GPUS=4
NUM_NODES=1
NUM_WORKERS=4
RUN_NAME="public-400epochs"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter --slurm_array_job_id $SLURM_ARRAY_JOB_ID --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --region $REGION --patch_size $PATCH_SIZE --num_epochs $NUM_EPOCHS --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_workers $NUM_WORKERS --use_logger $USE_LOGGER
