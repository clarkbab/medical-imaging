#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-05:00:00
#SBATCH -A punim1413

module load python/3.8.6
source ~/venvs/medical-imaging/bin/activate
module load web_proxy

version=$(python --version)
echo $version

join () {
  local IFS="$1"
  shift
  echo "$*"
}

DATASET="PMCC-HN-TRAIN"
DEST_DATASET="$DATASET-MULTI-HALF"
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
ROUND_DP=None
SIZE=None
OUTPUT_SPACING="(4,4,4)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
REGIONS=$(join , "${REGIONS[@]}")

python $SCRIPT_DIR/process/nifti/convert_to_training \
  --dataset $DATASET \
  --regions $REGIONS \
  --dest_dataset $DEST_DATASET \
  --output_spacing $OUTPUT_SPACING
  --round_dp $ROUND_DP \
  --size $SIZE \
