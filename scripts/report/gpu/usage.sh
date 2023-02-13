#! /usr/bin/env bash
module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

NAME="memory-test"
TIME=20
INTERVAL=0.1
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/gpu/usage.py \
    --name $NAME \
    --time $TIME \
    --interval $INTERVAL
