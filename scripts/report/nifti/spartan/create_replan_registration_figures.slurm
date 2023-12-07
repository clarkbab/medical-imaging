#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 32G
#SBATCH --time 0-04:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
CENTRE_OF="['Brain','Brainstem','Cavity_Oral','Bone_Mandible']"
REGION=
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_registration_figures.py \
    --dataset $DATASET \
    --centre_of $CENTRE_OF
