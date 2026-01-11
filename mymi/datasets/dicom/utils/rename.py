import os
import pydicom as dcm
import shutil
from tqdm import tqdm

from mymi import logging
from mymi.typing import *
from mymi.utils import *

from ..dataset import DicomDataset

def get_new_pat_id(
    old_pat: PatientID,
    rename_fn: Union[Callable, Dict[str, str]],
    pat_regexp: Optional[str] = None) -> PatientID:
    if pat_regexp is not None:
        match = re.match(pat_regexp, old_pat)
        if match is None:
            return old_pat
    if isinstance(rename_fn, Callable):
        return rename_fn(old_pat)
    elif old_pat in rename_fn:
        return rename_fn[old_pat]
    else:
        return old_pat

def rename_dicom(
    filepath: FilePath,
    rename_fn: Union[Callable, Dict[str, str]],
    makeitso: bool = False,
    pat_regexp: Optional[str] = None) -> None:
    # Get new patient ID.
    dicom = dcm.dcmread(filepath)
    old_pat = dicom.PatientID
    new_pat = get_new_pat_id(old_pat, rename_fn, pat_regexp=pat_regexp)
    if old_pat == new_pat:
        return

    if not makeitso:
        logging.info(f"Rename patient ID from {old_pat} to {new_pat} in {filepath}.")
    else:
        dicom.PatientID = new_pat
        dicom.PatientName = new_pat
        dicom.save_as(filepath)

def rename_patients(
    dataset: DatasetID,
    rename_fn: Optional[Callable],
    makeitso: bool = False,
    pat: PatientIDs = 'all',
    pat_regexp: Optional[str] = None) -> None:
    # Check if indexes are open and therefore can't be overwritten.
    dset = DicomDataset(dataset)
    files = ['index.csv', 'index-errors.csv']
    assert_writeable(files)

    # Rename all DICOMs in the index.
    logging.info("Renaming all indexed dicoms.")
    index = dset.index()
    filepaths = index['filepath'].tolist()
    for f in tqdm(filepaths):
        f = os.path.join(dset.path, 'data', 'patients', f)
        rename_dicom(f, rename_fn, makeitso=makeitso, pat_regexp=pat_regexp)

    # Rename all DICOMs in the error index.
    logging.info("Renaming all indexed dicoms with errors.")
    index_errors = dset.index_errors()
    filepaths = index_errors['filepath'].tolist()
    for f in tqdm(filepaths):
        f = os.path.join(dset.path, 'data', 'patients', f)
        rename_dicom(f, rename_fn, makeitso=makeitso, pat_regexp=pat_regexp)

    # Updating index filepaths.
    if pat_regexp is None:
        logging.warning("No patient ID regexp provided. Skipping renaming of filepaths.")
    else:
        logging.info("Renaming index filepaths.")
        src_filepaths = index['filepath'].tolist()
        dest_filepaths = []
        for f in src_filepaths:
            # Check full filepath for match.
            match = re.match(pat_regexp, f)
            if match is not None:
                old_pat_id = match.group(0)
                # Get new patient ID.
                new_pat_id = get_new_pat_id(old_pat_id, rename_fn, pat_regexp=pat_regexp)
                if new_pat_id == old_pat_id:
                    dest_filepaths.append(f)
                    continue

                # Get new filepath.
                new_filepath = re.sub(pat_regexp, new_pat_id, f)
                dest_filepaths.append(new_filepath)

                # Rename filepath.
                srcpath = os.path.join(dset.path, 'data', 'patients', f)
                destpath = os.path.join(dset.path, 'data', 'patients', new_filepath)
                if not makeitso:
                    logging.info(f"Moving file from {srcpath} to {destpath}.")
                else:
                    if not os.path.exists(os.path.dirname(destpath)):
                        os.makedirs(os.path.dirname(destpath), exist_ok=True)
                    shutil.move(srcpath, destpath)

                    # Remove old empty directories.
                    deleted = set()
                    for current_dir, subdirs, files in os.walk(srcpath, topdown=False):
                        still_has_subdirs = False
                        for subdir in subdirs:
                            if os.path.join(current_dir, subdir) not in deleted:
                                still_has_subdirs = True
                                break
                        if not any(files) and not still_has_subdirs:
                            os.rmdir(current_dir)
                            deleted.add(current_dir)
            else:
                dest_filepaths.append(f)
        index['filepath'] = dest_filepaths

        # Updating error index filepaths.
        logging.info("Renaming error index filepaths.")
        src_filepaths = index['filepath'].tolist()
        dest_filepaths = []
        for f in src_filepaths:
            # Check full filepath for match.
            match = re.match(pat_regexp, f)
            if match is not None:
                old_pat_id = match.group(0)
                # Get new patient ID.
                new_pat_id = get_new_pat_id(old_pat_id, rename_fn, pat_regexp=pat_regexp)
                if new_pat_id == old_pat_id:
                    dest_filepaths.append(f)
                    continue

                # Get new filepath.
                new_filepath = re.sub(pat_regexp, new_pat_id, f)
                dest_filepaths.append(new_filepath)

                # Rename filepath.
                srcpath = os.path.join(dset.path, 'data', 'patients', f)
                destpath = os.path.join(dset.path, 'data', 'patients', new_filepath)
                if not makeitso:
                    logging.info(f"Moving file from {srcpath} to {destpath}.")
                else:
                    if not os.path.exists(os.path.dirname(destpath)):
                        os.makedirs(os.path.dirname(destpath), exist_ok=True)
                    shutil.move(srcpath, destpath)

                    # Remove old empty directories.
                    deleted = set()
                    for current_dir, subdirs, files in os.walk(srcpath, topdown=False):
                        still_has_subdirs = False
                        for subdir in subdirs:
                            if os.path.join(current_dir, subdir) not in deleted:
                                still_has_subdirs = True
                                break
                        if not any(files) and not still_has_subdirs:
                            os.rmdir(current_dir)
                            deleted.add(current_dir)
            else:
                dest_filepaths.append(f)
        index['filepath'] = dest_filepaths

    # Update patient IDs in index.
    logging.info("Renaming patient IDs in index.")
    for i, r in index.iterrows():
        old_pat_id = r['patient-id']
        new_pat_id = get_new_pat_id(old_pat_id, rename_fn, pat_regexp=pat_regexp)
        if not makeitso:
            logging.info(f"Rename patient ID from {old_pat_id} to {new_pat_id} in index.")
        else:
            index.at[i, 'patient-id'] = new_pat_id
            index.at[i, 'patient-name'] = new_pat_id

    # Update patient IDs in error index.
    logging.info("Renaming patient IDs in error index.")
    for i, r in index_errors.iterrows():
        old_pat_id = r['patient-id']
        new_pat_id = get_new_pat_id(old_pat_id, rename_fn, pat_regexp=pat_regexp)
        if not makeitso:
            logging.info(f"Rename patient ID from {old_pat_id} to {new_pat_id} in error index.")
        else:
            index_errors.at[i, 'patient-id'] = new_pat_id
            index_errors.at[i, 'patient-name'] = new_pat_id

    # Save indexes.
    if makeitso:
        filepath = os.path.join(dset.path, 'index.csv')
        save_csv(index, filepath, index=True)
        filepath = os.path.join(dset.path, 'index-errors.csv')
        save_csv(index_errors, filepath, index=True)

    # Rename reports.
    logging.info("Renaming patient IDs in reports.")
    dirpath = os.path.join(dset.path, 'data', 'reports')
    if os.path.exists(dirpath):
        files = os.listdir(dirpath)
        for f in files:
            if f.endswith('.csv'):
                filepath = os.path.join(dirpath, f)
                df = load_csv(filepath)
                for i, r in df.iterrows():
                    old_pat_id = r['patient-id']
                    if pat_regexp is not None:
                        match = re.match(pat_regexp, old_pat_id)
                        if match is None:
                            continue
                    new_pat_id = rename_fn(old_pat_id)
                    if not makeitso:
                        logging.info(f"Rename patient ID from {old_pat_id} to {new_pat_id} in report {f}.")
                    else:
                        df.at[i, 'patient-id'] = new_pat_id
                        df.at[i, 'patient-name'] = new_pat_id
                if makeitso:
                    save_csv(df, filepath)
