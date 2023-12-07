#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition cascade
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00
#SBATCH -A punim1413

module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASETS="('PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC')" # Code links from 'training' set to nifti set.
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
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
# METRICS="('apl-mm-tol-1','dice','dm-surface-dice-tol-1','hd','hd-95','msd')"
# MODEL_TYPES="('clinical','transfer')"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
# STATS="('mean','min','max')"

# Remove 'quotes' from variables passed via bulk script - we need to pass 'quoted' strings to 'sbatch'.
METRICS=$(echo $METRICS | tr -d '"')
MODEL_TYPES=$(echo $MODEL_TYPES | tr -d '"')
STATS=$(echo $STATS | tr -d '"')

python $SCRIPT_DIR/transfer-learning/create_bootstrap_samples_and_predictions.py \
    --dataset $DATASETS \
    --region $REGION \
    --model_type $MODEL_TYPES \
    --metric $METRICS \
    --stat $STATS
