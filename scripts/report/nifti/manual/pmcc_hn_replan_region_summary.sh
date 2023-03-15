#! /usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")"; cd .. ;pwd -P )
cd $parent_path

DATASET="PMCC-HN-REPLAN"
REGION='Brain'
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_region_summary.py \
    --dataset $DATASET \
    --region $REGION

