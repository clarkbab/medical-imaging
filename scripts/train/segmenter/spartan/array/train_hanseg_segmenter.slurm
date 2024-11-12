#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
##SBATCH --mem 128G
#SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASETS="HAN-SEG-112"
REGIONS=(
    'A_Carotid_L'       # 0
    'A_Carotid_R'       # 1 
    'Arytenoid'         # 2
    'Bone_Mandible'     # 3
    'Brainstem'         # 4
    'BuccalMucosa'      # 5
    'Cavity_Oral'       # 6
    'Cochlea_L'         # 7
    'Cochlea_R'         # 8
    'Cricopharyngeus'   # 9
    'Esophagus_S'       # 10
    'Eye_AL'            # 11
    'Eye_AR'            # 12
    'Eye_PL'            # 13
    'Eye_PR'            # 14
    'Glnd_Lacrimal_L'   # 15
    'Glnd_Lacrimal_R'   # 16
    'Glnd_Submand_L'    # 17
    'Glnd_Submand_R'    # 18
    'Glnd_Thyroid'      # 19
    'Glottis'           # 20
    'Larynx_SG'         # 21
    'Lips'              # 22
    'OpticChiasm'       # 23
    'OpticNrv_L'        # 24
    'OpticNrv_R'        # 25
    'Parotid_L'         # 26
    'Parotid_R'         # 27
    'Pituitary'         # 28
    'SpinalCord'        # 29
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-hanseg-$REGION"
N_EPOCHS=150
N_FOLDS=None
N_GPUS=1
N_NODES=1
N_TRAIN=None
N_WORKERS=45
PRETRAINED_MODEL=None
RESUME=False
RESUME_RUN=None
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=None
RUN_NAME="epochs-150"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter/train.py \
    --datasets $DATASETS \
    --region $REGION \
    --model $MODEL_NAME \
    --run $RUN_NAME \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --pretrained_model $PRETRAINED_MODEL \
    --use_logger $USE_LOGGER \
    --resume $RESUME \
    --resume_run $RESUME_RUN \
    --resume_ckpt $RESUME_CKPT \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD
