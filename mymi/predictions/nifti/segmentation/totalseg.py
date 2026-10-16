from dicomset import NiftiDataset
from dicomset.utils import load_nifti, save_nifti, load_nrrd, regions_to_list, logger
from dicomset.typing import *
import numpy as np
import os
import shutil
import subprocess
import tempfile
from tqdm import tqdm
from typing import *

def create_totalseg_predictions(
    dataset: str,
    # Currently only totalseg groupings are supported, e.g. 'lung'.
    # See options: https://github.com/wasserth/TotalSegmentator/blob/211c2bd73386a0a48847d70678315bb326dbcd54/totalsegmentator/bin/totalseg_combine_masks.py#L34
    combine_regions: Dict[str, RegionID] = {},
    ct_series: SeriesID = 'series_0',
    makeitso: bool = False,
    overwrite_labels: bool = False,
    patient_ids: PatientID | List[PatientID] | Literal['all'] = 'all',
    regions_series: SeriesID = 'series_0',
    rename_regions: Union[Dict[RegionID, RegionID], Callable[[RegionID], RegionID]] = {},
    save_as_labels: bool = False,
    study_ids: StudyID | List[StudyID] | Literal['all'] = 'all',
    task_regions: Dict[str, Union[str, List[str], Literal['all']]] = {},
    ) -> None:
    roi_subset_tasks = ['total', 'total_mr']

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(patient_ids=patient_ids)

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        pat_studys = pat.list_studies(study_ids=study_ids)
        for s in tqdm(pat_studys, leave=False):
            study = pat.study(s)
            input_series = study.ct_series(ct_series)

            # Either save as predictions or labels.
            if save_as_labels:
                save_dir = os.path.join(set.path, 'data', 'patients', p, s, 'regions', regions_series)
            else:
                save_dir = os.path.join(set.path, 'data', 'predictions', 'segmentation', p, s, regions_series, 'totalseg')

            # Write totalseg predictions to a temporary directory. This is because we don't (necessarily) want all of the
            # predictions and we can copy just those we do want. We can't limit the predicted masks to a subset of
            # structures using '--roi_subset' except for 'total/total_mr' tasks.
            with tempfile.TemporaryDirectory() as temp_dir:
                for task, regions in task_regions.items():
                    regions = regions_to_list(regions, literals={'all': 'all'})
                    output_dir = os.path.join(temp_dir, task)
                    os.makedirs(output_dir, exist_ok=True)

                    # Create temp nifti files if '.nrrd' files are present.
                    # Totalseg doesn't handle nrrd.
                    input_filepath = input_series.filepath
                    if input_filepath.endswith('.nrrd'):
                        d, a = load_nrrd(input_filepath)
                        dest_filepath = os.path.join(output_dir, input_filepath.split('/')[-1].replace('.nrrd', '.nii.gz'))
                        logger.info(f"Copying NRRD to NIFTI: {input_filepath} -> {dest_filepath}")
                        save_nifti(d, a, dest_filepath)
                        input_filepath = dest_filepath

                    # Convert data from LPS to RAS, as required by totalseg.
                    # Copy to a temp file so we don't modify the actual input data on disk.
                    d, a = load_nifti(input_filepath)
                    d = np.flip(d, axis=(0, 1))  # Flip x and y axes.
                    temp_input_filepath = os.path.join(output_dir, 'input_ras.nii.gz')
                    save_nifti(d, a, temp_input_filepath)
                    input_filepath = temp_input_filepath

                    # Make total seg predictions.
                    command = [
                        'TotalSegmentator',
                        '--task', task,
                        '-i', input_filepath,
                        '-o', output_dir,
                        '--output_type', 'nifti',
                    ]
                    if 'TOTALSEG_LICENCE' in os.environ:
                        command += ['-l', os.environ['TOTALSEG_LICENCE']]
                    # Predict subset of regions if possible.
                    if task in roi_subset_tasks and regions != 'all':
                        command += ['--roi_subset'] + regions
                    logger.info(command)
                    subprocess.run(command)

                    # Remove temp input files.
                    if input_series.filepath.endswith('.nrrd'):
                        # Remove temp nifti file.
                        logger.info(f"Removing temporary NIFTI input file: {input_filepath}")
                        os.remove(input_filepath)
                    os.remove(temp_input_filepath) 

                    # Remove unwanted regions, if subset predictions wasn't possible.
                    print('regions:', regions)
                    if task not in roi_subset_tasks and regions != 'all':
                        files = os.listdir(output_dir)
                        print(files)
                        for f in files:
                            region = f.replace('.nii.gz', '')
                            if region not in regions:
                                filepath = os.path.join(output_dir, f)
                                logger.info(f"Removing {task}:{region}. Filepath: {filepath}")
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
                    #     logger.info(command)
                    #     subprocess.run(command)

                    # Rename regions.
                    if callable(rename_regions):
                        files = os.listdir(output_dir)
                        for f in files:
                            region = f.replace('.nii.gz', '')
                            oldpath = os.path.join(output_dir, f'{region}.nii.gz')
                            newname = rename_regions(region)
                            newpath = os.path.join(output_dir, f'{newname}.nii.gz')
                            logger.info(f'Renaming {oldpath} to {newpath}')
                            os.rename(oldpath, newpath)
                    elif isinstance(rename_regions, dict):
                        for o, n in rename_regions.items():
                            oldpath = os.path.join(output_dir, f'{o}.nii.gz')
                            if os.path.exists(oldpath):
                                newpath = os.path.join(output_dir, f'{n}.nii.gz')
                                logger.info(f'Renaming {oldpath} to {newpath}')
                                os.rename(oldpath, newpath)

                    # Convert from RAS to LPS.
                    files = os.listdir(output_dir)
                    for f in files:
                        filepath = os.path.join(output_dir, f)
                        d, a = load_nifti(filepath)
                        d = np.flip(d, axis=(0, 1))
                        save_nifti(d, a, filepath)

                    # Copy predictions to the output directory.
                    files = os.listdir(output_dir)
                    for f in files:
                        oldpath = os.path.join(output_dir, f)
                        newpath = os.path.join(save_dir, f)
                        logger.info(f'Copying {oldpath} to {newpath}')
                        if makeitso:
                            os.makedirs(os.path.dirname(newpath), exist_ok=True)
                            if save_as_labels and not overwrite_labels and os.path.exists(newpath):
                                raise ValueError(f"File {newpath} already exists. Use 'overwrite_labels=True' to overwrite existing labels.")
                            else:
                                shutil.copyfile(oldpath, newpath)
