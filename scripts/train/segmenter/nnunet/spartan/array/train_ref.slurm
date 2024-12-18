#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
##SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set nnUNet vars.
export nnUNet_raw="/data/projects/punim1413/mymi/datasets/nnunet/raw"
export nnUNet_preprocessed="/data/projects/punim1413/mymi/datasets/nnunet/preprocessed"
export nnUNet_results="/data/projects/punim1413/mymi/models/nnunet"
export nnUNet_keep_files_open=True

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
#TEST_FOLD=4
DATASET_ID=$((21 + $TEST_FOLD * $N_REGIONS + $SLURM_ARRAY_TASK_ID))
echo $DATASET_ID

MODEL="3d_fullres"
FOLD="all"
PLAN="nnUNetResEncUNetPlansXXL"
TRAINER="nnUNetTrainerNoMirroring"

# --c restarts from latest checkpoint.
command="nnUNetv2_train $DATASET_ID $MODEL $FOLD \
    -p $PLAN \
    -tr $TRAINER"
echo $command
eval $command
