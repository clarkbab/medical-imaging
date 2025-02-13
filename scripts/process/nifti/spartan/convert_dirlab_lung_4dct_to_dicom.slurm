#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 64G
##SBATCH --gres gpu:1
##SBATCH --cpus-per-gpu 8
#SBATCH --time 0-06:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

# Set other parameters.
DATASET="DIRLAB-LUNG-4DCT"
LANDMARKS_PREFIX="Marker"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

command="python $SCRIPT_DIR/process/nifti/convert_to_dicom.py \
    $DATASET \
    $DATASET \
    --landmarks_prefix $LANDMARKS_PREFIX"
echo $command
$command
