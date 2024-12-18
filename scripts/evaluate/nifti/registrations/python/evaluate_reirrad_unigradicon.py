import os
import subprocess
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
sys.path.append(root_dir)

from mymi.evaluation.dataset.nifti import create_registration_evaluation

dataset = 'PMCC-REIRRAD'
moving_study_id = 'study_0'
fixed_study_id = 'study_1'
landmarks = 'all'
regions = 'RL:PMCC-REIRRAD'
regions = None
model = 'UNIGRADICON'
create_registration_evaluation(dataset, moving_study_id, fixed_study_id, model, landmarks=landmarks, regions=regions)
