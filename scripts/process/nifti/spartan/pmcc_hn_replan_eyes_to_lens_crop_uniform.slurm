#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition sapphire
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

# Set parameters.
CROP_METHOD="lens-uniform"
CROP_MM="(330,380,500)"
DATASET="PMCC-HN-REPLAN"
DEST_DATASET="PMCC-HN-REPLAN-EYES-UNIFORM"
PAT_IDS="['62-0','94-1','73-0','69-0','52-0','43-0','87-1','51-1','59-0','9-0','43-1','73-1','103-1','57-0','57-1']"
REGION="['Eye_L','Eye_R','Lens_L','Lens_R']"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nifti/convert_replan_to_lens_crop.py \
    --dataset $DATASET \
    --pat_ids $PAT_IDS \
    --dest_dataset $DEST_DATASET \
    --crop_method $CROP_METHOD \
    --crop_mm $CROP_MM \
    --region $REGION
