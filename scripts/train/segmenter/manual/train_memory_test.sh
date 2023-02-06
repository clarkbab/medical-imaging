#! /usr/bin/env bash
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
python --version

DATASETS="['PMCC-HN-TEST-MULTI','PMCC-HN-TRAIN-MULTI']"
MODEL_NAME="memory-test"
RUN_NAME="memory-test"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/train/segmenter/train_memory_test.py \
    --dataset $DATASETS \
    --model $MODEL_NAME \
    --run $RUN_NAME
