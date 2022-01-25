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

DATASET="HNSCC"
REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Brain'             # 2
    'BrainStem'         # 3
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'Mandible'          # 8
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
    'OralCavity'        # 11
    'Parotid_L'         # 12
    'Parotid_R'         # 13
    'SpinalCord'        # 14
    'Submandibular_L'   # 15
    'Submandibular_R'   # 16
)
DILATE_REGIONS=(
    'BrachialPlexus_L'  # 0
    'BrachialPlexus_R'  # 1
    'Cochlea_L'         # 4
    'Cochlea_R'         # 5
    'Lens_L'            # 6
    'Lens_R'            # 7
    'OpticNerve_L'      # 9
    'OpticNerve_R'      # 10
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
DEST_DATASET="$DATASET-$REGION-LOC"
DILATE=False
if [[ " ${DILATE_REGIONS[*]} " =~ " ${REGION} " ]]; then
    DILATE=True
fi
ROUND_DP=2
SIZE="(128,128,150)"
SPACING="(4,4,4)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/process/nifti/convert_to_training --dataset $DATASET --region $REGION --dest_dataset $DEST_DATASET --dilate $DILATE \
    --round_dp $ROUND_DP --size $SIZE --spacing $SPACING
