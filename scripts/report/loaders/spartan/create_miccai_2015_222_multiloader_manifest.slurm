#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:10:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

. scripts/utilities/bash_functions.sh

DATASET="MICCAI-2015-222"
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
N_FOLDS=None
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_SPLIT_FILE

python $SCRIPT_DIR/report/loaders/create_multi_loader_manifest.py \
    --dataset $DATASET \
    --region $REGION \
    --n_folds $N_FOLDS \
    --use_split_file $USE_SPLIT_FILE