#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 64G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 2-0:00:00
#SBATCH -A punim1413
#SBATCH -q gpgpumse

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASETS="['PMCC-HN-TEST-SEG','PMCC-HN-TRAIN-SEG']"
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
MODEL_NAME="segmenter-${REGION}"
NUM_EPOCHS=150
NUM_FOLDS=5
NUM_GPUS=1
NUM_NODES=1
# NUM_TRAIN=5
NUM_WORKERS=4
# RESUME=False
RESUME_CHECKPOINT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
# TEST_FOLD=0
RUN_NAME="clinical-fold-$TEST_FOLD-samples-$NUM_TRAIN"
USE_LOGGER=True

python $SCRIPT_DIR/train/segmenter \
    --datasets $DATASETS \
    --model_name $MODEL_NAME \
    --num_epochs $NUM_EPOCHS \
    --num_folds $NUM_FOLDS \
    --num_gpus $NUM_GPUS \
    --num_nodes $NUM_NODES \
    --num_train $NUM_TRAIN \
    --num_workers $NUM_WORKERS \
    --use_logger $USE_LOGGER \
    --region $REGION \
    --resume $RESUME \
    --resume_checkpoint $RESUME_CHECKPOINT \
    --run_name $RUN_NAME \
    --slurm_array_job_id $SLURM_ARRAY_JOB_ID \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    --test_fold $TEST_FOLD
