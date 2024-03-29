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

DATASET="PMCC-HN-TEST"
REGIONS=(
    'Bone_Mandible'     # 0
    'BrachialPlex_L'    # 1
    'BrachialPlex_R'    # 2
    'Brain'             # 3
    'Brainstem'         # 4
    'Cavity_Oral'       # 5
    'Cochlea_L'         # 6
    'Cochlea_R'         # 7
    'Esophagus_S'       # 8
    'Eye_L'             # 9
    'Eye_R'             # 10
    'Glnd_Submand_L'    # 11
    'Glnd_Submand_R'    # 12
    'Glottis'           # 13
    'GTVp'              # 14
    'Larynx'            # 15
    'Lens_L'            # 16
    'Lens_R'            # 17
    'Musc_Constrict'    # 18
    'OpticChiasm'       # 19
    'OpticNrv_L'        # 20
    'OpticNrv_R'        # 21
    'Parotid_L'         # 22
    'Parotid_R'         # 23
    'SpinalCord'        # 24
)
REGION=$(join "${REGIONS[@]}")
OUTPUT_SPACING="(1,1,2)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TRAINING_DATASET="$DATASET-112"

python $SCRIPT_DIR/process/nifti/convert_to_training.py \
    --dataset $DATASET \
    --region $REGION \
    --output_spacing $OUTPUT_SPACING \
    --training_dataset $TRAINING_DATASET
