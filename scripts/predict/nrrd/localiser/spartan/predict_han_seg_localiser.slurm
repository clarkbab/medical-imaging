#!/bin/bash
#SBATCH --account punim1413
#SBATCH --qos feit
#SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
##SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 120G
#SBATCH --time 0-04:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

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
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
LOCALISER="('localiser-hanseg-$REGION','epochs-150','BEST')"
N_EPOCHS=150
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TIMING=True

python $SCRIPT_DIR/predict/nrrd/localiser/create_all_predictions.py \
    --dataset $DATASET \
    --localiser $LOCALISER \
    --n_epochs $N_EPOCHS \
    --timing $TIMING 