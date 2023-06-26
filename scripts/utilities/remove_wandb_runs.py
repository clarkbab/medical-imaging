from fire import Fire
import os
import shutil
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from mymi import config

dry_run = False

# Get runs.
filepath = os.path.join('scripts', 'utilities', 'wandb_runs.txt')
runs = open(filepath, 'r').readlines()

# Remove trailing newlines.
runs = [r.rstrip() for r in runs]

for run in runs:
    filepath = os.path.join(config.directories.reports, run)
    if not os.path.exists(filepath):
        continue

    if dry_run:
        print(f"Removing run '{run}'. Filepath: '{filepath}'.")
    else:
        print(f"Removing run '{run}'. Filepath: '{filepath}'.")
        try:
            shutil.rmtree(filepath)
        except OSError as e:
            print(f"Caught exception: '{str(e)}'.")
            if 'OSError: [Errno 39] Directory not empty: ' in str(e):
                print(f"Retrying removal of run '{run}'. Filepath: '{filepath}'.")
            else:
                raise e
