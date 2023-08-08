import os
from os.path import dirname as up
import pathlib
import re
import sys
from tqdm import tqdm

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config

IMAGE_REGEXP = r'^desc:([\w-]+):([\w]+):region:([\w]+):axis:([0-2])_([\d]+)_[\w]+.png$'
MASK_REGEXP = r'^desc:([\w-]+):([\w]+):region:([\w]+):axis:([0-2])_([\d]+)_[\w]+.mask.png$'
RUN_REGEXP = r'^run-\d{8}_\d{6}-[\da-z]{8}$'

dry_run = False
first_n_epochs = 1000
steps_per_epoch = 30
first_n_steps = first_n_epochs * steps_per_epoch

run_folder = os.path.join(config.directories.reports, 'wandb')
runs = os.listdir(run_folder)
runs = [r for r in runs if re.match(RUN_REGEXP, r) is not None]

for run in runs:
    image_folder = os.path.join(run_folder, run, 'files', 'media', 'images')
    if not os.path.exists(image_folder):
        # Images may have been cleared out by wandb.
        continue

    # Remove images.
    images = os.listdir(image_folder)
    for image in images:
        match = re.match(IMAGE_REGEXP, image)
        if match is None:
            # Could be 'mask' folder.
            continue
        
        image_step = int(match.group(5))
        if image_step < first_n_steps:
            image_path = os.path.join(image_folder, image)
            if dry_run:
                print(f"Removing image '{image}'. Filepath: '{image_path}'.")
            else:
                print(f"Removing image '{image}'. Filepath: '{image_path}'.")
                os.remove(image_path)
            
    # Remove masks.
    mask_folder = os.path.join(image_folder, 'mask')
    if not os.path.exists(mask_folder):
        continue
    masks = os.listdir(mask_folder)
    for mask in masks:
        match = re.match(MASK_REGEXP, mask)
        if match is None:
            continue

        mask_step = int(match.group(5))
        if mask_step < first_n_steps:
            mask_path = os.path.join(mask_folder, mask)
            if dry_run:
                print(f"Removing mask '{mask}'. Filepath: '{mask_path}'.")
            else:
                print(f"Removing mask '{mask}'. Filepath: '{mask_path}'.")
                os.remove(mask_path)

