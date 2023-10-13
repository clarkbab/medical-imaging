#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 128G
#SBATCH --time 0-06:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="['PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC']"
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="localiser-$REGION"
#TEST_FOLD=0
#N_TRAIN=None
RUN_NAME="clinical-fold-$TEST_FOLD-samples-$N_TRAIN"
#N_EPOCHS=150
N_FOLDS=5
N_GPUS=1
N_NODES=1
N_WORKERS=8
#RESUME=False
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOGGER=True

python $SCRIPT_DIR/train/localiser/train_localiser.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --run_name $RUN_NAME \
    --region $REGION \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_gpus $N_GPUS \
    --n_nodes $N_NODES \
    --n_train $N_TRAIN \
    --n_workers $N_WORKERS \
    --resume $RESUME \
    --resume_ckpt $RESUME_CKPT \
    --test_fold $TEST_FOLD \
    --use_logger $USE_LOGGER
