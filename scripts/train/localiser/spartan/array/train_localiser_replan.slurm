#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition feit-gpu-a100
#SBATCH --mem 32G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 0-12:00:00
#SBATCH -A punim1413
#SBATCH -q feit 

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-LOC"
REGIONS=(
    'Bone_Mandible'
    'BrachialPlex_L'
    'BrachialPlex_R'
    'Brain'
    'Brainstem'
    'Cavity_Oral'
    'Cochlea_L'
    'Cochlea_R'
    'Esophagus_S'
    'Eye_L'
    'Eye_R'
    'Glnd_Submand_L'
    'Glnd_Submand_R'
    'Glottis'
    'GTVp'
    'Larynx'
    'Lens_L'
    'Lens_R'
    'Musc_Constrict'
    'OpticChiasm'
    'OpticNrv_L'
    'OpticNrv_R'
    'Parotid_L'
    'Parotid_R'
    'SpinalCord'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="localiser-replan-$REGION"
TEST_FOLD=0
HALVE_CHANNELS=True
RUN_NAME="multi-class-$TEST_FOLD"
LR_FIND=False
N_EPOCHS=300
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser/train_localiser_replan.py \
    --dataset $DATASET \
    --region $REGION \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --halve_channels $HALVE_CHANNELS \
    --lr_find $LR_FIND \
    --n_epochs $N_EPOCHS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
