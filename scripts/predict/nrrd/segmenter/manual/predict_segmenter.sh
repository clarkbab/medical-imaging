#! /usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd ..; pwd -P )
cd $parent_path

REGION='Bone_Mandible'
SHORT_REGION='BM'
MODEL_NAME="segmenter-miccai"
RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
RUN_NAME="1-region-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
N_SPLIT_CHANNELS=2
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/predict/nrrd/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --n_split_channels $N_SPLIT_CHANNELS \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
