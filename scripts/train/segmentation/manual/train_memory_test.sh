#! /usr/bin/env bash
module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
python --version

# CKPT_LIBRARY="baseline"
# CKPT_LIBRARY="ckpt-pytorch"
CKPT_LIBRARY="ckpt-fairscale"
# CKPT_LIBRARY="ckpt-fairscale-offload"
CKPT_MODE=""
# CKPT_MODE="-level"
INPUT_MODE="-large"
# INPUT_MODE=""
# INPUT_MODE="-small"
# INPUT_MODE="-xsmall"
MONITOR_TIME=60
N_CKPTS=30
N_TRAIN_STEPS=6
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/train/segmenter/train_memory_test.py \
    --ckpt_library=$CKPT_LIBRARY \
    --ckpt_mode=$CKPT_MODE \
    --input_mode=$INPUT_MODE \
    --n_ckpts=$N_CKPTS \
    --monitor_time=$MONITOR_TIME \
    --n_train_steps=$N_TRAIN_STEPS
