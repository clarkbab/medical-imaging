#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-1:00:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DATASET="HN1"
REGIONS=(
    'Brain'             # 0
    'Cochlea_L'         # 1
    'Cochlea_R'         # 2
    'OralCavity'        # 3
    'Parotid_L'         # 4
    'Parotid_R'         # 5
    'SpinalCord'        # 6
    'Submandibular_L'   # 7
    'Submandibular_R'   # 8
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_region_summary --dataset $DATASET --region $REGION
