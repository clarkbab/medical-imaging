#!/bin/bash
#SBATCH --account punim1413
#SBATCH --partition sapphire
##SBATCH --partition gpu-a100-short
##SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

DATASET="PMCC-HN-REPLAN"
REGION="['Bone_Mandible','BrachialPlex_L','BrachialPlex_R','Brain','Brainstem','Cavity_Oral','Cochlea_L','Cochlea_R','Esophagus_S','Eye_L','Eye_R','GTVp','Glnd_Submand_L','Glnd_Submand_R','Glottis','Larynx','Lens_L','Lens_R','Musc_Constrict','OpticChiasm','OpticNrv_L','OpticNrv_R','Parotid_L','Parotid_R','SpinalCord']"
LOAD_ALL_SAMPLES=True
N_FOLDS=5
TEST_FOLDS=(0 1 2 3 4)
TEST_FOLD=${TEST_FOLDS[$SLURM_ARRAY_TASK_ID]}
USE_GROUPING=True
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

python $SCRIPT_DIR/evaluate/nifti/registration/evaluate_registration.py \
    --dataset $DATASET \
    --region $REGION \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_folds $N_FOLDS \
    --test_fold $TEST_FOLD \
    --use_grouping $USE_GROUPING
