#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 64G
##SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    'Bone_Mandible'     # 0
    'Brainstem'         # 1
    'Glnd_Submand_L'    # 2
    'Glnd_Submand_R'    # 3
    'OpticChiasm'       # 4
    'OpticNrv_L'        # 5
    'OpticNrv_R'        # 6
    'Parotid_L'         # 7
    'Parotid_R'         # 8
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}

FIRST_DATASET_ID=106
SPACING="(1,1,2)"
DATASET="MICCAI-CROP"
DATASET_ID=$(($FIRST_DATASET_ID + $SLURM_ARRAY_TASK_ID))
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
echo $DATASET_ID

command="python $SCRIPT_DIR/process/nnunet/convert_predictions_to_nifti_single_region.py \
    --dataset $DATASET \
    --dataset_id $DATASET_ID \
    --region $REGION \
    --spacing $SPACING"
echo $command
$command
