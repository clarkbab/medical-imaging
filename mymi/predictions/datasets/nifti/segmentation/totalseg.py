import subprocess
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

def create_totalseg_predictions(
    dataset: str,
    combine_regions: Dict[str, str] = {},
    pat_ids: PatientIDs = 'all',
    remove_task_regions: Optional[Union[str, Dict[str, Union[str, List[str]]], Literal['all']]] = None,
    task_regions: Dict[str, Union[str, List[str], Literal['all']]] = {},
    save_as_labels: bool = False,
    splits: Splits = 'all',
    study_ids: StudyIDs = 'all') -> None:

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        pat_study_ids = pat.list_studies(study_ids=study_ids)
        for s in tqdm(pat_study_ids, leave=False):
            study = pat.study(s)

            # Make total seg prediction.
            if save_as_labels:
                output_path = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
            else:
                output_path = os.path.join(set.path, 'data', 'predictions', 'segmentation', p, s, 'totalseg')
            for task, regions in task_regions.items():
                command = [
                    'TotalSegmentator',
                    '-i', study.ct_path,
                    '-o', output_path,
                    '--task', task,
                ]
                if regions != 'all':
                    regions = regions_to_list(regions)
                    command += ['--roi_subset'] + regions
                logging.info(command)
                subprocess.run(command)

            # Combine regions.
            for c, r in combine_regions.items():
                output_filepath = os.path.join(output_path, f'{r}.nii.gz')
                command = [
                    'totalseg_combine_masks',
                    '-i', output_path,
                    '-o', output_filepath,
                    '-m', c
                ]
                logging.info(command)
                subprocess.run(command)

            # Remove specified regions.
            for task, regions in task_regions.items():
                if remove_task_regions == 'all':
                    regions = regions_to_list(regions)
                    for r in regions:
                        filepath = os.path.join(output_path, f'{r}.nii.gz')
                        os.remove(filepath)
