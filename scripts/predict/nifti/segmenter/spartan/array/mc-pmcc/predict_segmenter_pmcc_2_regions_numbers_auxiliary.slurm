#!/bin/bash
#SBATCH --account punim1413
##SBATCH --qos feit
##SBATCH --partition feit-gpu-a100
##SBATCH --partition gpu-a100
#SBATCH --partition gpu-a100-short
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-00:30:00
##SBATCH --test-only

module load GCCcore/11.3.0
module load Python/3.10.4
source ~/venvs/medical-imaging/bin/activate

version=$(python --version)
echo $version

REGIONS=(
    "['OralCavity']"                                        # 0
    "['Brain','OralCavity']"                                # 1
    "['BrainStem','OralCavity']"                            # 2
    "['OralCavity','Mandible']"                             # 3
    "['Brain','OralCavity','BrainStem','Mandible']"         # 4

    "['SpinalCord']"                                        # 5
    "['SpinalCord','BrainStem']"                            # 6
    "['Mandible','SpinalCord']"                             # 7
    "['OralCavity','SpinalCord']"                           # 8
    "['SpinalCord','Submandibular_L']"                      # 9
    "['SpinalCord','Submandibular_R']"                      # 10
    "['SpinalCord','BrainStem','Parotid_L','Parotid_R']"    # 11
    "['SpinalCord','Mandible','BrainStem','OralCavity','Parotid_L','Parotid_R','Submandibular_L','Submandibular_R']"    # 12

    "['BrachialPlexus_L']"                                  # 13
    "['BrachialPlexus_L','BrachialPlexus_R']"               # 14
    "['BrachialPlexus_L','Cochlea_L']"                      # 15
    "['BrachialPlexus_L','Cochlea_R']"                      # 16
    "['BrachialPlexus_L','Lens_L']"                         # 17
    "['BrachialPlexus_L','Lens_R']"                         # 18
    "['BrachialPlexus_L','OpticNerve_L']"                   # 19
    "['BrachialPlexus_L','OpticNerve_R']"                   # 20
    "['BrachialPlexus_L','Cochlea_L','Cochlea_R']"          # 21
    "['BrachialPlexus_L','Lens_L','Lens_R']"                # 22
    "['BrachialPlexus_L','OpticNerve_L','OpticNerve_R']"    # 23
    "['BrachialPlexus_L','BrachialPlexus_R','OpticNerve_L','OpticNerve_R']"    # 24

    "['BrachialPlexus_R']"                                  # 25
    "['BrachialPlexus_R','Cochlea_L']"                      # 26
    "['BrachialPlexus_R','Cochlea_R']"                      # 27
    "['BrachialPlexus_R','Lens_L']"                         # 28
    "['BrachialPlexus_R','Lens_R']"                         # 29
    "['BrachialPlexus_R','OpticNerve_L']"                   # 30
    "['BrachialPlexus_R','OpticNerve_R']"                   # 31
    "['BrachialPlexus_R','Cochlea_L','Cochlea_R']"          # 32
    "['BrachialPlexus_R','Lens_L','Lens_R']"                # 33
    "['BrachialPlexus_R','OpticNerve_L','OpticNerve_R']"    # 34
    
    "['Cochlea_L']"                                         # 35
    "['Cochlea_L','Lens_L']"                                # 36
    "['Cochlea_L','Lens_R']"                                # 37
    "['Cochlea_L','Cochlea_R','Lens_L','Lens_R']"           # 38

    "['Cochlea_R']"                                         # 39
    "['Cochlea_R','Lens_L']"                                # 40
    "['Cochlea_R','Lens_R']"                                # 41
    
    "['Lens_L']"                                            # 42

    "['Lens_R']"                                            # 43

    # Does the mandible help everything?
    "['Mandible','BrachialPlexus_L']"                       # 44
    "['Mandible','BrachialPlexus_R']"                       # 45
    "['Mandible','Cochlea_L']"                              # 46
    "['Mandible','Cochlea_R']"                              # 47
    "['Mandible','Lens_L']"                                 # 48
    "['Mandible','Lens_R']"                                 # 49
)
SHORT_REGIONS=(
    'OC'
    'B_OC'
    'BS_OC'
    'OC_M'  
    'B_OC_BS_M'

    'SC'
    'SC_BS'
    'M_SC'  
    'OC_S'  
    'SC_SL'  
    'SC_SR'  
    'SC_4'
    'SC_8'

    'BL'
    'BL_BR'
    'BL_CL'
    'BL_CR'
    'BL_LL'
    'BL_LR'
    'BL_OL'
    'BL_OR'
    'BL_CL_CR'
    'BL_LL_LR'
    'BL_OL_OR'
    'BL_BR_OL_OR'

    'BR'
    'BR_CL'
    'BR_CR'
    'BR_LL'
    'BR_LR'
    'BR_OL'
    'BR_OR'
    'BR_CL_CR'
    'BR_LL_LR'
    'BR_OL_OR'

    'CL'
    'CL_LL'
    'CL_LR'
    'CL_CR_LL_LR'

    'CR'
    'CR_LL'
    'CR_LR'

    'LL'

    'LR'

    'M_BL'
    'M_BR'
    'M_CL'
    'M_CR'
    'M_LL'
    'M_LR'
)
REGION=${REGIONS[$SLURM_ARRAY_TASK_ID]}
SHORT_REGION=${SHORT_REGIONS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="segmenter-pmcc-numbers-v3"
RESOLUTION="112"
DATASET="('PMCC-HN-TEST-BC-$RESOLUTION','PMCC-HN-TRAIN-BC-$RESOLUTION')"
RANDOM_SEED=45
RUN_NAME_BASE="2-regions-$SHORT_REGION-$RESOLUTION-seed-$RANDOM_SEED"
if [ "$USE_WEIGHTS" = True ]; then
    RUN_NAME="$RUN_NAME_BASE-ivw-$WEIGHTS_IV_FACTOR_0-schedule-$SCHEDULE_EPOCHS"
elif [ "$SHORT_REGION" = "LL_LR" ] || [ "$SHORT_REGION" = "CL_LL" ] || [ "$SHORT_REGION" = "CL_LR" ] || [ "$SHORT_REGION" = "CR_LL" ] || [ "$SHORT_REGION" = "CR_LR" ]; then
    LR_INIT=1e-4
    RUN_NAME="$RUN_NAME_BASE-lr-$LR_INIT"
else
    RUN_NAME="$RUN_NAME_BASE"
fi
MODEL="('$MODEL_NAME','$RUN_NAME','BEST')"
CHECK_EPOCHS=True
CROP_MM="(300,400,500)"
N_EPOCHS=1000
N_FOLDS=5
N_SPLIT_CHANNELS=2
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"
TEST_FOLD=0
USE_LOADER_SPLIT_FILE=False
LOAD_ALL_SAMPLES=True

python $SCRIPT_DIR/predict/nifti/segmenter/predict_multi.py \
    --dataset $DATASET \
    --region $REGION \
    --model $MODEL \
    --check_epochs $CHECK_EPOCHS \
    --crop_mm $CROP_MM \
    --load_all_samples $LOAD_ALL_SAMPLES \
    --n_epochs $N_EPOCHS \
    --n_folds $N_FOLDS \
    --n_split_channels $N_SPLIT_CHANNELS \
    --test_fold $TEST_FOLD \
    --use_loader_split_file $USE_LOADER_SPLIT_FILE
