from os.path import dirname as up
import pathlib
import sys

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(up(filepath)))))
sys.path.append(mymi_dir)
from mymi.processing.dataset.nifti import convert_segmenter_predictions_to_dicom

DATASETS = ('PMCC-HN-TEST-LOC','PMCC-HN-TRAIN-LOC') # Code links from 'training' set to nifti set.
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
NUM_FOLDS = 5
TEST_FOLDS = (0, 1, 2, 3, 4)
NUM_TRAINS = (5, 10, 20, 50, 100, None)
MODELS = ['clinical', 'public', 'transfer']

regions = []
localisers = []
segmenters = []
for region in REGIONS:
    for model in MODELS:
        for test_fold in TEST_FOLDS:
            if model == 'public':
                regions += region
                localisers += (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')
                segmenters += (f'segmenter-{region}', 'public-1gpu-150epochs', 'BEST')
            else:
                for num_train in NUM_TRAINS:
                    regions += region
                    localisers += (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')
                    segmenters += (f'segmenter-{region}', f'{model}-fold-{test_fold}-samples-{num_train}', 'BEST')

for test_fold in TEST_FOLDS:
    convert_segmenter_predictions_to_dicom_from_loader(DATASETS, regions, localisers, segmenters, num_folds=NUM_FOLDS, test_fold=test_fold)
