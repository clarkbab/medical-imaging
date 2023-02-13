#! /usr/bin/env bash
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
python --version

MODE="baseline"
# MODE="ckpt-pytorch"
# MODE="ckpt-fairscale"
# MODE="ckpt-fairscale-offload"
MONITOR_TIME=30
N_CKPTS=1
NAME="memory-test-small-$MODE-n_ckpts-$N_CKPTS"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/train/segmenter/train_memory_test.py \
    --mode $MODE \
    --name $NAME \
    --monitor_time $MONITOR_TIME \
    --n_ckpts $N_CKPTS
