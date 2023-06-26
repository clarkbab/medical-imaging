#! /usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd ..; pwd -P )
cd $parent_path

RESOLUTION="222"
DATASET="MICCAI-2015-$RESOLUTION"
REGION="Brainstem"
SHORT_REGION="BS"
MODEL_NAME="segmenter-miccai-no-background"
RUN_NAME="1-region-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','best')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
CHECK_EPOCHS=True
N_SPLIT_CHANNELS=2
USE_LOADER_SPLIT_FILE=TRUE

python $SCRIPT_DIR/predict/nrrd/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --n_split_channels $N_SPLIT_CHANNELS \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
