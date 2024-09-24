#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 128G
##SBATCH --mem 120G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 7-00:00:00
##SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

IMGPATH="/data/gpfs/projects/punim1413/mymi/datasets/nifti/L2R-LUNG-CT-CROP-TRAIN/voxelmorph-index.csv"
MODELPATH="/data/gpfs/projects/punim1413/mymi/models/voxelmorph/L2R_LUNG_CT_CROP"
VMXPATH="/home/baclark/code/voxelmorph"

command="python $VMXPATH/scripts/torch/train.py --gpu 0 --img-list $IMGPATH --model-dir $MODELPATH"
echo $command
eval $command
