#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 32G
#SBATCH --time 0-04:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
CENTRE="['Brain','Brainstem','Cavity_Oral','Bone_Mandible']"
CROP="['Brain','Brainstem','Cavity_Oral','Bone_Mandible']"
CROP_MARGIN=150
REGION="['Bone_Mandible','BrachialPlex_L','BrachialPlex_R','Brain','Brainstem','Cavity_Oral','Cochlea_L','Cochlea_R','Esophagus_S','Eye_L','Eye_R','GTVp','Glnd_Submand_L','Glnd_Submand_R','Glottis','Larynx','Lens_L','Lens_R','Musc_Constrict','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R','SpinalCord']"
SHOW_LEGEND=False
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_registration_figures.py \
    --dataset $DATASET \
    --centre $CENTRE \
    --crop $CROP \
    --crop_margin $CROP_MARGIN \
    --region $REGION \
    --show_legend $SHOW_LEGEND
