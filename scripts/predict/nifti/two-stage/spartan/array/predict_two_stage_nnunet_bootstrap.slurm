#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASETS="PMCC-HN-REPLAN-112"
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
INST_REGIONS=( 
    'Mandible'     # 0
    'BrachialPlexus_L'    # 1
    'BrachialPlexus_R'    # 2
    'Brain'             # 3
    'BrainStem'         # 4
    'OralCavity'       # 5
    'Esophagus_S'       # 6
    'GTVp'              # 7
    'Submandibular_L'    # 8
    'Submandibular_R'    # 9
    'Larynx'            # 10
    'Lens_L'            # 11
    'Lens_R'            # 12
    'Musc_Constrict'    # 13
    'Parotid_L'         # 14
    'Parotid_R'         # 15
    'SpinalCord'        # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
INST_REGION=${INST_REGIONS[$SLURM_ARRAY_TASK_ID]}
LOCALISER="('localiser-$INST_REGION','public-1gpu-150epochs','BEST')"
SEGMENTER="('segmenter-$INST_REGION-v2','clinical-fold-0-samples-None','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
N_EPOCHS=150
N_FOLDS=5
TEST_FOLD=0

python $SCRIPT_DIR/predict/nifti/two-stage/create_predictions.py \
    --datasets $DATASETS \
    --region $REGION \
    --localiser $LOCALISER \
    --segmenter $SEGMENTER \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD
