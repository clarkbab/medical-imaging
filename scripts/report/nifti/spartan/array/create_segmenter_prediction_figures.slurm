#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-04:00:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

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
LOCALISER="('localiser-$REGION','public-1gpu-150epochs','BEST')"
# MODEL='clinical'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
# TEST_FOLD=0
# if [ "$MODEL" = "public" ]; then
#     run_name="public-1gpu-150epochs"
# else
#     run_name="$MODEL-fold-$TEST_FOLD-samples-$N_TRAIN"
# fi
# SEGMENTER="('segmenter-$REGION','$run_name','BEST')"

# For 'SpinalCord' caudal extent check.
read -r -d '' SEGMENTER << EOF
[
    ('segmenter-SpinalCord', 'public-1gpu-150epochs', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-5', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-10', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-20', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-50', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-100', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-200', 'best'),
    ('segmenter-SpinalCord-v2', 'clinical-fold-$TEST_FOLD-samples-None', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-5', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-10', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-20', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-50', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-100', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-200', 'best'),
    ('segmenter-SpinalCord-v2', 'transfer-fold-$TEST_FOLD-samples-None', 'best')
]
EOF

SEGMENTER=$(echo "$SEGMENTER" | tr -d " \t\n\r" )     # Remove whitespace.
VIEW="sagittal"

python $SCRIPT_DIR/report/nifti/create_segmenter_prediction_figures.py \
    --dataset $DATASETS \
    --region $REGION \
    --localiser $LOCALISER \
    --segmenter $SEGMENTER \
    --test_fold $TEST_FOLD \
    --view $VIEW
