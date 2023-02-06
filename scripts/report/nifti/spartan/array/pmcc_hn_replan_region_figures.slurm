#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-6:00:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET='PMCC-HN-REPLAN'
REGIONS=(
    'Bone_Mandible'
    'BrachialPlex_L'
    'BrachialPlex_R'
    'Brain'
    'Brainstem'
    'Cavity_Oral'
    'Cochlea_L'
    'Cochlea_R'
    'Esophagus_S'
    'Eye_L'
    'Eye_R'
    'Glnd_Submand_L'
    'Glnd_Submand_R'
    'Glottis'
    'GTVp'
    'Larynx'
    'Lens_L'
    'Lens_R'
    'Musc_Constrict'
    'OpticChiasm'
    'OpticNrv_L'
    'OpticNrv_R'
    'Parotid_L'
    'Parotid_R'
    'SpinalCord'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
SUBREGIONS=False

python $SCRIPT_DIR/report/nifti/create_region_figures.py \
    --dataset $DATASET \
    --region $REGION \
    --subregions $SUBREGIONS
