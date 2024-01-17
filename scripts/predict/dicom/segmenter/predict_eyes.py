import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.prediction.dataset.dicom import create_all_multi_segmenter_predictions

# # Set path to data.
# if not os.path.exists('data'):
#     os.makedirs('data')
# os.environ['MYMI_DATA'] = '.'

print(os.environ['MYMI_DATA'])

# Get path to input DICOM files.
dataset = sys.argv[1]
regions = ['Eye_L', 'Eye_R', 'Lens_L', 'Lens_R']
model = ('segmenter-replan-eyes', '4-regions-EL_ER_LL_LR-112-seed-42-lr-1e-4', 'loss=-0.096713-epoch=671-step=0')
model_spacing = (1, 1, 2)
check_epochs = True
n_epochs = 1000

create_all_multi_segmenter_predictions(dataset, regions, model, model_spacing, check_epochs=check_epochs, n_epochs=n_epochs)
