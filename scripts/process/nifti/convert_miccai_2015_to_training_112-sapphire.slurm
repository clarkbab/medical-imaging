#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
#SBATCH --nodes 1
#SBATCH --mem 128G
#SBATCH --time 0-06:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

python convert_miccai_2015_to_training_112.py 
