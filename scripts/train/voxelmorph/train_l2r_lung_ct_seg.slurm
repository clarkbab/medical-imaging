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
module load TensorFlow/2.13.0-CUDA-11.8.0
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

IMGPATH="/data/gpfs/projects/punim1413/mymi/datasets/nifti/L2R-LUNG-CT-CROP-TRAIN/voxelmorph-index-seg.csv"
IMGPREFIX="/data/gpfs/projects/punim1413/mymi/datasets/nifti/L2R-LUNG-CT-CROP-TRAIN/data/ct/"
SEGPREFIX="/data/gpfs/projects/punim1413/mymi/datasets/nifti/L2R-LUNG-CT-CROP-TRAIN/data/regions/Lung/"
LABELPATH="/data/gpfs/projects/punim1413/mymi/datasets/nifti/L2R-LUNG-CT-CROP-TRAIN/voxelmorph-index-seg-labels.npy"
MODELPATH="/data/gpfs/projects/punim1413/mymi/models/voxelmorph/L2R_LUNG_CT_CROP_SEG"
VMXPATH="/home/baclark/code/voxelmorph"

command="python $VMXPATH/scripts/tf/train_semisupervised_seg.py \
    --gpu 0 \
    --image-loss "mse" \
    --img-list $IMGPATH \
    --img-prefix $IMGPREFIX \
    --seg-prefix $SEGPREFIX \
    --img-suffix ".nii.gz" \
    --seg-suffix ".nii.gz" \
    --labels $LABELPATH \
    --model-dir $MODELPATH"
echo $command
eval $command
