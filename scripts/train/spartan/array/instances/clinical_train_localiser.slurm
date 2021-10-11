#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 32G
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

DATASETS="PMCC-HN-TRAIN-LOC"
BASH_REGIONS=('BrachialPlexus_L' 'BrachialPlexus_R' 'Brain' 'BrainStem' 'Cochlea_L' 'Cochlea_R' 'Lens_L' 'Lens_R' 'Mandible' 'OpticChiasm' 'OpticNerve_L' 'OpticNerve_R' 'OralCavity' 'Parotid_L' 'Parotid_R' 'SpinalCord' 'Submandibular_L' 'Submandibular_R')
REGIONS="['BrachialPlexus_L','BrachialPlexus_R','Brain','BrainStem','Cochlea_L','Cochlea_R','Lens_L','Lens_R','Mandible','OpticChiasm','OpticNerve_L','OpticNerve_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R']"
REGION=${BASH_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="localiser-${REGION}"
NUM_EPOCHS=400
NUM_GPUS=4
NUM_NODES=1
NUM_WORKERS=4
RUN_NAME="pmcc-hn-train-400epochs"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser --slurm_job_id $SLURM_JOB_ID --slurm_array_task_id $SLURM_ARRAY_TASK_ID --model_name $MODEL_NAME --run_name $RUN_NAME --datasets $DATASETS --num_epochs $NUM_EPOCHS --num_gpus $NUM_GPUS --num_nodes $NUM_NODES --num_workers $NUM_WORKERS --region $REGION --use_logger $USE_LOGGER
