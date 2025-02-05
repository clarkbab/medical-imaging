#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set nnUNet vars.
export nnUNet_raw_data_base="/data/projects/punim1413/mymi/datasets/nnunet/v1/raw"
# export nnUNet_predictions="/data/projects/punim1413/mymi/datasets/nnunet/v1/predictions"
export nnUNet_preprocessed="/data/projects/punim1413/mymi/datasets/nnunet/v1/preprocessed"
export RESULTS_FOLDER="/data/projects/punim1413/mymi/models/nnunet/v1"

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
FIRST_TASK_ID=001
TASK_ID=$(($FIRST_TASK_ID + $SLURM_ARRAY_TASK_ID))
PLANNER="ExperimentPlanner3D_v21"

command="""mednextv1_plan_and_preprocess \
    -t $TASK_ID \
    -pl3d $PLANNER \
    --verify_dataset_integrity"""
echo $command
$command
