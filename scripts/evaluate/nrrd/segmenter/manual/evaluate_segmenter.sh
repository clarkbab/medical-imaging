#! /usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd ..; pwd -P )
cd $parent_path

RESOLUTION="444"
DATASET="MICCAI-2015-$RESOLUTION"
REGION='Bone_Mandible'
SHORT_REGION='BM'
MODEL_NAME="segmenter-miccai"
RUN_NAME="1-region-$SHORT_REGION-$RESOLUTION"
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
USE_LOADER_SPLIT_FILE=True

python $SCRIPT_DIR/evaluate/nrrd/segmenter/evaluate.py \
    --dataset $DATASET \
    --model $MODEL \
    --model_region $REGION \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
