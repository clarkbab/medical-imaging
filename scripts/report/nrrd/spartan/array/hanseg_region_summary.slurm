#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition sapphire
#SBATCH --mem 32G
#SBATCH --time 0-6:00:00
#SBATCH -A punim1413

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
DATASET="HAN-SEG"
REGIONS=(
    'A_Carotid_L'       # 0
    'A_Carotid_R'       # 1 
    'Arytenoid'         # 2
    'Bone_Mandible'     # 3
    'Brainstem'         # 4
    'BuccalMucosa'      # 5
    'Cavity_Oral'       # 6
    'Cochlea_L'         # 7
    'Cochlea_R'         # 8
    'Cricopharyngeus'   # 9
    'Esophagus_S'       # 10
    'Eye_AL'            # 11
    'Eye_AR'            # 12
    'Eye_PL'            # 13
    'Eye_PR'            # 14
    'Glnd_Lacrimal_L'   # 15
    'Glnd_Lacrimal_R'   # 16
    'Glnd_Submand_L'    # 17
    'Glnd_Submand_R'    # 18
    'Glnd_Thyroid'      # 19
    'Glottis'           # 20
    'Larynx_SG'         # 21
    'Lips'              # 22
    'OpticChiasm'       # 23
    'OpticNrv_L'        # 24
    'OpticNrv_R'        # 25
    'Parotid_L'         # 26
    'Parotid_R'         # 27
    'Pituitary'         # 28
    'SpinalCord'        # 29
)
REGIONS=${REGIONS[$SLURM_ARRAY_TASK_ID]}

python $SCRIPT_DIR/report/nrrd/create_region_summary.py \
    --dataset $DATASET \
    --regions $REGIONS
