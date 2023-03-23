#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-06:00:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

DATASET="MICCAI-2015"
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
OUTPUT_SPACING="(1,1,2)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TRAINING_DATASET="$DATASET-112"

python $SCRIPT_DIR/process/nrrd/convert_to_training.py \
    --dataset $DATASET \
    --region $REGION \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET
