#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition gpgpu
#SBATCH --mem 64G
#SBATCH --gres gpu:p100:1
#SBATCH --cpus-per-gpu 4
#SBATCH --time 0-06:00:00
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
MODEL_NAME="segmenter-${REGION}-v2"
# MODEL="clinical"
# N_EPOCHS=150
N_FOLDS=5
N_GPUS=1
N_NODES=1
# N_TRAIN=5
N_WORKERS=4
# RESUME=False
RESUME_RUN=None
RESUME_CKPT='last'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
# TEST_FOLD=0
if [ "$MODEL" = "public" ]; then
    RUN_NAME="public-1gpu-150epochs"
else
    RUN_NAME="$MODEL-fold-$TEST_FOLD-samples-$N_TRAIN"
fi
if [ "$MODEL" = "transfer" ]; then
    PRETRAINED_MODEL="('segmenter-$REGION','public-1gpu-150epochs','BEST')"
else
    PRETRAINED_MODEL=None
fi
USE_LOGGER=True

# Check if 'N_TRAIN' training samples exist for this region.
N_TRAIN_MAX=$(python $SCRIPT_DIR/utilities/get_n_train_max.py \
    --datasets $DATASETS \
    --region $REGION \
    --test_fold $TEST_FOLD)

if [ \( "$N_TRAIN" == "all" \) -o \( "$N_TRAIN" -ge "$N_TRAIN_MAX" \) ]; then
    echo "N_TRAIN ($N_TRAIN) >= N_TRAIN_MAX ($N_TRAIN_MAX). Skipping."
    exit 0
fi

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
