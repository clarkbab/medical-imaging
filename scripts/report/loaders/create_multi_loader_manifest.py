import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.reports.loaders import create_multi_loader_manifest

if __name__ == '__main__':
    fire.Fire(create_multi_loader_manifest)
