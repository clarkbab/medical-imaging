#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 1-0:00:00
#SBATCH -A punim1413

module load gcccore/8.3.0
module load python/3.8.2
module load web_proxy

version=$(python --version)
echo $version

DEST_DATASET="HN1-SEG"
REGIONS="('Brain','Cochlea_L','Cochlea_R','MedullaOblongata','Neck_L','Neck_R','OralCavity','Parotid_L','Parotid_R','SpinalCord','Submandibular_L','Submandibular_R')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Launch a job for each region.
bash $SCRIPT_DIR/process/hn1_seg $REGIONS $DEST_DATASET