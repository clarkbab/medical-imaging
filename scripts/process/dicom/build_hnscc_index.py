import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import datasets as ds

ds.get('HNSCC', 'dicom').list_patients()
