#!/bin/bash

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET='PMCC-HN-REPLAN'
REGION='Brain'
SUBREGIONS=False
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_region_figures.py \
    --dataset $DATASET \
    --region $REGION \
    --subregions $SUBREGIONS
