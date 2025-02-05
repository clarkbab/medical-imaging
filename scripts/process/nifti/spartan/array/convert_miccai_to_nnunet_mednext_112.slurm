#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 64G
##SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-06:00:00

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

# Set other parameters.
DATASET="MICCAI-CROP"
FIRST_DATASET_ID=1
N_VAL=8
NORMALISE=True
NORM_MEAN=-840.04
NORM_STDEV=383.73
SHUFFLE_TRAIN=True
SPACING="(1,1,2)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nifti/convert_to_nnunet_single_region_v1.py \
    --dataset $DATASET \
    --first_dataset_id $FIRST_DATASET_ID \
    --region $REGION \
    --n_val $N_VAL \
    --normalise $NORMALISE \
    --norm_mean $NORM_MEAN \
    --norm_stdev $NORM_STDEV \
    --shuffle_train $SHUFFLE_TRAIN \
    --spacing $SPACING \
