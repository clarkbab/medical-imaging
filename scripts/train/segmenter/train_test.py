import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.training import train_segmenter_test

if __name__ == '__main__':
    fire.Fire(train_segmenter_test)

# Sample args:
# --datasets "['HN1-SEG','HNSCC-SEG']" --num_gpus 4 --num_nodes 1 --num_workers 4 --use_logger True
