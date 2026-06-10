from dicomset.utils import logger, sort_lists
import numpy as np
import os
import shutil
from tqdm.auto import tqdm

from mymi.utils.cdog import list_fractions, get_kv_path

patpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient04"
fractions = [4, 5]
makeitso = True

for f in tqdm(fractions, desc='Fractions'):
    # Get the frame splits.
    fraction_path = os.path.join(patpath, f"Fx{f:02d}")
    dirpath = os.path.join(fraction_path, 'kV')
    # Get kV folder.
    kv_dirpath = get_kv_path(fraction_path)
    tiff_files = [f for f in os.listdir(kv_dirpath) if f.endswith('.tiff')]
    arcs = [int(f.split('_')[1]) for f in tiff_files]
    frames = [int(f.split('_')[2]) for f in tiff_files]
    tiff_files, arcs, frames = sort_lists([tiff_files, arcs, frames], key=lambda taf: taf[2])

    # Get non-consecutive frames.
    arcs_diff = np.diff(arcs)
    frames_diff = np.diff(frames)

    # Assert that arcs jump where frames jump.
    jump_idxs = np.argwhere(frames_diff != 1).flatten()

    # Get frame splits.
    frame_splits = [frames[i] + 1 for i in jump_idxs] + [frames[-1] + 1]
    print('frame idxs: ', jump_idxs)
    print('frame numbers: ', frame_splits)
    print('frames missing: ', frames_diff[jump_idxs])

    if not makeitso:
        continue

    # Copy data to new folder.
    logger.info(f"Copying data for fraction {f} to new folder...")
    new_kvpath = os.path.join(fraction_path, 'kV_new')
    if os.path.exists(new_kvpath):
        shutil.rmtree(new_kvpath)
    shutil.copytree(kv_dirpath, new_kvpath)

    # Rename files in the new folder.
    logger.info(f"Renaming files for fraction {f}...")
    tiff_files = list(sorted([f for f in os.listdir(new_kvpath) if f.endswith('.tiff')]))
    old_arcs = [int(f.split('_')[1]) for f in tiff_files]
    frames = [int(f.split('_')[2]) for f in tiff_files]
    for oa, fr, tf in tqdm(zip(old_arcs, frames, tiff_files), total=len(frames), leave=False, desc='Frames'):
        oldpath = os.path.join(new_kvpath, tf)
        newpath = oldpath
        for i, s in enumerate(frame_splits):
            arc = i + 1
            if fr < s:
                newpath = os.path.join(new_kvpath, tf.replace(f'Ch1_{oa}', f'Ch1_{arc}'))
                break
        if oldpath != newpath:
            shutil.move(oldpath, newpath)
