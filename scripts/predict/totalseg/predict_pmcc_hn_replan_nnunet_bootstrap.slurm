#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
##SBATCH --mem 128G
#SBATCH --mem 120G
#SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
##SBATCH --time 1-00:00:00
#SBATCH --time 2-00:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
#TASKS="['total','head_glands_cavities','brain_structures','headneck_bones_vessels','headneck_muscles']"
TASKS="['brain_structures']"

command="python $MYMI_CODE/scripts/predict/totalseg/predict.py \
    --dataset $DATASET \
    --tasks $TASKS"
echo $command
eval $command
