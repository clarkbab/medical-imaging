import shutil
import subprocess
import tempfile
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

def create_totalseg_predictions(
    dataset: str,
    # Currently only totalseg groupings are supported, e.g. 'lung'.
    # See options: https://github.com/wasserth/TotalSegmentator/blob/211c2bd73386a0a48847d70678315bb326dbcd54/totalsegmentator/bin/totalseg_combine_masks.py#L34
    combine_regions: Dict[str, RegionID] = {},
    dry_run: bool = True,
    overwrite_labels: bool = False,
    pat_ids: PatientIDs = 'all',
    rename_regions: Union[Dict[RegionID, RegionID], Callable[[RegionID], RegionID]] = {},
    task_regions: Dict[str, Union[str, List[str], Literal['all']]] = {},
    save_as_labels: bool = False,
    splits: Splits = 'all',
    study_ids: StudyIDs = 'all') -> None:
    roi_subset_tasks = ['total', 'total_mr']

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        pat_study_ids = pat.list_studies(study_ids=study_ids)
        for s in tqdm(pat_study_ids, leave=False):
            study = pat.study(s)

            # Either save as predictions or labels.
            if save_as_labels:
                save_dir = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
            else:
                save_dir = os.path.join(set.path, 'data', 'predictions', 'segmentation', p, s, 'totalseg')

            # Write totalseg predictions to a temporary directory. This is because we don't (necessarily) want all of the
            # predictions and we can copy just those we do want. We can't limit the predicted masks to a subset of
            # structures using '--roi_subset' except for 'total/total_mr' tasks.
            with tempfile.TemporaryDirectory() as temp_dir:
                for task, region_ids in task_regions.items():
                    region_ids = regions_to_list(region_ids, literals={'all': 'all'})
                    output_dir = os.path.join(temp_dir, task)
                    os.makedirs(output_dir, exist_ok=True)

                    # Create temp nifti files if '.nrrd' files are present.
                    # Totalseg doesn't handle nrrd.
                    input_filepath = study.ct_filepath
                    if input_filepath.endswith('.nrrd'):
                        d, s, o = load_nrrd(input_filepath)
                        dest_filepath = os.path.join(output_dir, input_filepath.split('/')[-1].replace('.nrrd', '.nii.gz'))
                        logging.info(f"Copying NRRD to NIFTI: {input_filepath} -> {dest_filepath}")
                        save_nifti(d, dest_filepath, spacing=s, offset=o)
                        input_filepath = dest_filepath

                    # Convert data from LPS to RAS, as required by totalseg.
                    d, s, o = load_nifti(input_filepath)
                    d = np.flip(d, axis=(0, 1))  # Flip x and y axes.
                    save_nifti(d, input_filepath, spacing=s, offset=o)

                    # Make total seg predictions.
                    command = [
                        'TotalSegmentator',
                        '--task', task,
                        '-i', input_filepath,
                        '-o', output_dir,
                        '--output_type', 'nifti',
                    ]
                    # Predict subset of regions if possible.
                    if task in roi_subset_tasks and region_ids != 'all':
                        command += ['--roi_subset'] + region_ids
                    logging.info(command)
                    subprocess.run(command)

                    if study.ct_filepath.endswith('.nrrd'):
                        # Remove temp nifti file.
                        logging.info(f"Removing temporary NIFTI input file: {input_filepath}")
                        os.remove(input_filepath)

                    # Remove unwanted regions, if subset predictions wasn't possible.
                    if task not in roi_subset_tasks and region_ids != 'all':
                        files = os.listdir(output_dir)
                        print(files)
                        for f in files:
                            region_id = f.replace('.nii.gz', '')
                            if region_id not in region_ids:
                                filepath = os.path.join(output_dir, f)
                                logging.info(f"Removing {task}:{region_id}. Filepath: {filepath}")
                                os.remove(filepath)

                    # # Combine regions.
                    # for c, r in combine_regions.items():
                    #     filepath = os.path.join(output_dir, f'{r}.nii.gz')
                    #     command = [
                    #         'totalseg_combine_masks',
                    #         '-i', output_dir,
                    #         '-o', filepath,
                    #         '-m', c
                    #     ]
                    #     logging.info(command)
                    #     subprocess.run(command)

                    # Rename regions.
                    if callable(rename_regions):
                        files = os.listdir(output_dir)
                        for f in files:
                            region_id = f.replace('.nii.gz', '')
                            oldpath = os.path.join(output_dir, f'{region_id}.nii.gz')
                            newname = rename_regions(region_id)
                            newpath = os.path.join(output_dir, f'{newname}.nii.gz')
                            logging.info(f'Renaming {oldpath} to {newpath}')
                            os.rename(oldpath, newpath)
                    elif isinstance(rename_regions, dict):
                        for o, n in rename_regions.items():
                            oldpath = os.path.join(output_dir, f'{o}.nii.gz')
                            if os.path.exists(oldpath):
                                newpath = os.path.join(output_dir, f'{n}.nii.gz')
                                logging.info(f'Renaming {oldpath} to {newpath}')
                                os.rename(oldpath, newpath)

                    # Convert from RAS to LPS.
                    files = os.listdir(output_dir)
                    for f in files:
                        filepath = os.path.join(output_dir, f)
                        d, s, o = load_nifti(filepath)
                        d = np.flip(d, axis=(0, 1))
                        save_nifti(d, filepath, spacing=s, offset=o)

                    # Copy predictions to the output directory.
                    files = os.listdir(output_dir)
                    for f in files:
                        oldpath = os.path.join(output_dir, f)
                        newpath = os.path.join(save_dir, f)
                        logging.info(f'Copying {oldpath} to {newpath}')
                        if not dry_run:
                            os.makedirs(os.path.dirname(newpath), exist_ok=True)
                            if save_as_labels and not overwrite_labels and os.path.exists(newpath):
                                raise ValueError(f"File {newpath} already exists. Use 'overwrite_labels=True' to overwrite existing labels.")
                            else:
                                shutil.copyfile(oldpath, newpath)
