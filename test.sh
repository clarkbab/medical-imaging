TEST_FOLD=0

read -r -d '' SEGMENTER << EOF
[
    ('segmenter-SpinalCord', 'public-1gpu-150epochs', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-5', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-10', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-20', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-50', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-100', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-200', 'best'),
    ('segmenter-SpinalCord', 'clinical-fold-$TEST_FOLD-samples-None', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-5', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-10', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-20', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-50', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-100', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-200', 'best'),
    ('segmenter-SpinalCord', 'transfer-fold-$TEST_FOLD-samples-None', 'best')
]
EOF

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

DATASETS="['PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC']"
REGION="SpinalCord"
LOCALISER="('localiser-$REGION', 'public-1gpu-150epochs', 'best')"

LOCALISER=$(echo "$LOCALISER" | tr -d " \t\n\r" )     # Remove whitespace.
SEGMENTER=$(echo "$SEGMENTER" | tr -d " \t\n\r" )     # Remove whitespace.

echo $LOCALISER
echo $SEGMENTER

python $SCRIPT_DIR/report/nifti/create_segmenter_prediction_figures.py \
    --dataset $DATASETS \
    --region $REGION \
    --localiser $LOCALISER \
    --segmenter $SEGMENTER \
    --test_fold $TEST_FOLD
