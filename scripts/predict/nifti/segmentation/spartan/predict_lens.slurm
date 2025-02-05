#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-00:30:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

#DATASET="HNPCT"
#PAT_ID="HN-CHUM-049"
DATASET="HNSCC"
PAT_ID="HNSCC-01-0134"
MODEL="('segmenter-pmcc-numbers-v3','2-regions-LL_LR-112-seed-48-lr-1e-4','best')"
MODEL_REGION="['Lens_L','Lens_R']"
MODEL_SPACING="(1,1,2)"
CHECK_EPOCHS=True
N_EPOCHS=1000
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/predict/nifti/segmenter/create_patient_multi_segmenter_prediction.py \
    --dataset $DATASET \
    --pat_id $PAT_ID \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --model_spacing $MODEL_SPACING \
    --check_epochs $CHECK_EPOCHS \
    --n_epochs $N_EPOCHS
