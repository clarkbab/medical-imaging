#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --time 0-06:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN-112"
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
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
# TEST_FOLD=0
MODEL="('segmenter-replan-112','n-folds-5-fold-$TEST_FOLD-ivw-0.5-schedule-200-seed-42')"
MODEL_REGION="RL:PMCC_REPLAN"
N_FOLDS=5
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_multi_segmenter_prediction_figures.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --model_region $MODEL_REGION \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD
