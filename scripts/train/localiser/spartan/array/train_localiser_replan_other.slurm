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

DATASET="PMCC-HN-REPLAN-OTHER-LOC"
REGIONS=(
    'Bone_Mandible'        # 0
    'BrachialPlex_L'       # 1
    'BrachialPlex_R'       # 2
    'Brain'                # 3
    'Brainstem'            # 4
    'Cavity_Oral'          # 5
    'Cochlea_L'            # 6
    'Cochlea_R'            # 7
    'Esophagus_S'          # 8
    'Eye_L'                # 9
    'Eye_R'                # 10
    'Glnd_Submand_L'       # 11
    'Glnd_Submand_R'       # 12
    'Glottis'              # 13
    'GTVp'                 # 14
    'Larynx'               # 15
    'Lens_L'               # 16
    'Lens_R'               # 17
    'Musc_Constrict'       # 18
    'OpticChiasm'          # 19
    'OpticNrv_L'           # 20
    'OpticNrv_R'           # 21
    'Parotid_L'            # 22
    'Parotid_R'            # 23
    'SpinalCord'           # 24
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="localiser-replan-$REGION"
TEST_FOLD=0
HALVE_CHANNELS=True
RUN_NAME="multi-class-halved-$TEST_FOLD"
LR_FIND=False
N_EPOCHS=200
N_GPUS=1
N_NODES=1
N_WORKERS=4
RESUME=False
RESUME_CHECKPOINT='last'
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
    --resume_checkpoint $RESUME_CHECKPOINT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
