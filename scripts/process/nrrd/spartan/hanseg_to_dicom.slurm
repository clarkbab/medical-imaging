#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
DATASET="HANSEG"

python $SCRIPT_DIR/process/nrrd/convert_to_dicom.py \
    --dataset $DATASET \
    --dest_dataset $DATASET \
    --pat_prefix $DATASET
