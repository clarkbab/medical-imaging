#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --mem 32G
#SBATCH --time 0-02:00:00
#SBATCH -A punim1413

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

join () {
  local IFS="$1"
  shift
  echo "$*"
}

DATASET="PMCC-HN-TEST"
DEST_DATASET="$DATASET-LOC-v2"
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
DILATE=False
if [[ " ${DILATE_REGIONS[*]} " =~ " ${REGION} " ]]; then
    DILATE=True
fi
ROUND_DP=None
SIZE="(128,128,150)"
SPACING="(4,4,4)"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
REGIONS=$(join , "${REGIONS[@]}")
DILATE_REGIONS=$(join , "${DILATE_REGIONS[@]}")

python $SCRIPT_DIR/process/nifti/convert_to_training.py \
    --dataset $DATASET \
    --dest_dataset $DEST_DATASET \
    --dilate_regions $DILATE_REGIONS \
    --region $REGIONS \
    --round_dp $ROUND_DP \
    --size $SIZE \
    --spacing $SPACING
