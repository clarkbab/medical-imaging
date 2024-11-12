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

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
REGIONS=(
    'Bone_Mandible'     # 0
    'BrachialPlex_L'    # 1
    'BrachialPlex_R'    # 2
    'Brain'             # 3
    'Brainstem'         # 4
    'Cavity_Oral'       # 5
    'Esophagus_S'       # 6
    'GTVp'              # 7
    'Glnd_Submand_L'    # 8
    'Glnd_Submand_R'    # 9
    'Larynx'            # 10
    'Lens_L'            # 11
    'Lens_R'            # 12
    'Musc_Constrict'    # 13
    'Parotid_L'         # 14
    'Parotid_R'         # 15
    'SpinalCord'        # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
N_REGIONS=17
TEST_FOLD=4

python $SCRIPT_DIR/process/nifti/convert_replan_to_nnunet_ref_model.py \
    --n_regions $N_REGIONS \
    --regions $REGION \
    --test_fold $TEST_FOLD
