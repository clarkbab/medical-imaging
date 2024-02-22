#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --mem 32G
#SBATCH --time 0-06:00:00
#SBATCH -A punim1413

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

DATASET="MICCAI-2015"
DEST_DATASET="$DATASET-BC-112"
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
REGION=$(join , "${REGIONS[@]}")
CROP_MM="[(-125,-175,-260),(120,55,30)]"
SPACING="(1,1,2)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nrrd/convert_brain_crop_to_training.py \
    --dataset $DATASET \
    --crop_mm $CROP_MM \
    --dest_dataset $DEST_DATASET \
    --region $REGION \
    --spacing $SPACING
