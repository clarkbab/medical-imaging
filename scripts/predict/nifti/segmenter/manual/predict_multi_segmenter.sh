#! /usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd ..; pwd -P )
cd $parent_path

DATASET="PMCC-HN-REPLAN"    # Loader links from 'training' to 'nifti' dataset using 'index.csv'.
REGIONS="all"
N_FOLDS=5
RESOLUTION="222"
MODEL="('segmenter-miccai','baseline-1-region-BM-$RESOLUTION','best')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGIONS \
    --model $MODEL \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
