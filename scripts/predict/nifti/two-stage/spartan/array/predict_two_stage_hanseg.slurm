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

DATASETS="('PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC')" # Code links from 'training' set to nifti set.
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
HANSEG_REGIONS=(
    'BrachialPlex_L'  # 0
    'BrachialPlex_R'  # 1
    'Brain'             # 2
    'Brainstem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Bone_Mandible'          # 8
    'OpticNrv_L'      # 9
    'OpticNrv_R'      # 10
    'Cavity_Oral'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Glnd_Submand_L'   # 15
    'Glnd_Submand_R'   # 16
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
HANSEG_REGION=${HANSEG_REGIONS[$SLURM_ARRAY_TASK_ID]}
LOCALISER="('localiser-hanseg-$HANSEG_REGION','epochs-150','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
SEGMENTER="('segmenter-hanseg-$HANSEG_REGION','epochs-150','BEST')"
# TEST_FOLD=0
N_EPOCHS=150

python $SCRIPT_DIR/predict/nifti/two-stage/create_predictions.py \
    --datasets $DATASETS \
    --region $REGION \
    --localiser $LOCALISER \
    --segmenter $SEGMENTER \
    --n_epochs $N_EPOCHS \
    --test_fold $TEST_FOLD
