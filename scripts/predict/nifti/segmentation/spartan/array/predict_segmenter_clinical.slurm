#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 32G
#SBATCH --time 0-00:30:00

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
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
# N_TRAIN=5
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
# TEST_FOLD=0
LOCALISER="('localiser-$REGION','clinical-fold-$TEST_FOLD-samples-$N_TRAIN','BEST')"
SEGMENTER="('segmenter-$REGION-v2','clinical-fold-$TEST_FOLD-samples-$N_TRAIN','BEST')"

python $SCRIPT_DIR/predict/nifti/segmenter/predict_v2.py \
    --datasets $DATASETS \
    --region $REGION \
    --localiser $LOCALISER \
    --segmenter $SEGMENTER \
    --n_train $N_TRAIN \
    --test_fold $TEST_FOLD
