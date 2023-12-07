import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.gradcam.dataset.nrrd import create_heatmap

dataset = "MICCAI-2015"
pat_id = "0522c0001"
model = ('segmenter-miccai-numbers', '1-region-BM-112-seed-42', 'best')
model_region = "Parotid_L"
model_spacing = (1, 1, 2)
region = "Parotid_L"
layer = ['5','12','19','26','33']
layer_spacing = [(1,1,2), (2,2,4), (4,4,8), (8,8,16), (16,16,32)]
check_epochs = True
n_epochs = 5000

create_heatmap(dataset, pat_id, model, model_region, model_spacing, region, layer, layer_spacing, check_epochs=check_epochs, n_epochs=n_epochs)
