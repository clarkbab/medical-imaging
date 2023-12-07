#!/bin/bash
#SBATCH --nodes 1
#SBATCH --partition physical
#SBATCH --mem 32G
#SBATCH --time 0-00:30:00
#SBATCH -A punim1413

module load gcccore/10.2.0
module load python/3.8.6
module load web_proxy
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
REGION="['Bone_Mandible','BrachialPlex_L','BrachialPlex_R','Brain','Brainstem','Cavity_Oral','Cochlea_L','Cochlea_R','Esophagus_S','Eye_L','Eye_R','GTVp','Glnd_Submand_L','Glnd_Submand_R','Glottis','Larynx','Lens_L','Lens_R','Musc_Constrict','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R','SpinalCord']"
CENTRE_OF="['Brain','Brainstem','Cavity_Oral','Bone_Mandible']"
CROP="['Brain','Brainstem','Cavity_Oral','Bone_Mandible']"
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/report/nifti/create_ct_summary.py \
    --dataset $DATASET \
    --centre_of $CENTRE_OF \
    --crop $CROP \
    --region $REGION
