import os
from os.path import dirname as up
import pathlib
import sys
import torch

filepath = pathlib.Path(__file__).resolve()
mymi_dir = up(up(up(up(filepath))))
sys.path.append(mymi_dir)
from mymi import config
from mymi.regions import RegionNames

dry_run = False

model_types = ['segmenter']
regions = RegionNames
model_methods = ['clinical', 'transfer']
folds = list(range(5))
# num_trains = [5, 10, 20]
num_trains = [5]
# num_epochses = [
#     [300, 450, 600, 750, 900],
#     [300, 450],
#     [300]
# ]
num_epochses = [
    [750, 900]
]

for model_type in model_types:
    for model_method in model_methods:
        for region in regions:
            for fold in folds:
                for num_epochs, num_train in zip(num_epochses, num_trains):
                    # Check that base run has finished.
                    model = f'{model_type}-{region}'
                    run = f'{model_method}-fold-{fold}-samples-{num_train}'
                    print(model, run)
                    ckptspath = os.path.join(config.directories.models, model, run)
                    lastpath = os.path.join(ckptspath, 'last.ckpt')
                    state = torch.load(lastpath, map_location=torch.device('cpu'))
                    epoch = state['epoch']
                    assert epoch >= 149

                    ckptspath_base = ckptspath
                    lastpath_base = lastpath
                    for num_epoch in num_epochs:
                        # Check that run has finished.
                        run = f'{model_method}-fold-{fold}-samples-{num_train}-{num_epoch}epochs'
                        ckptspath = os.path.join(config.directories.models, model, run)
                        lastpath = os.path.join(ckptspath, 'last.ckpt')
                        state = torch.load(lastpath, map_location=torch.device('cpu'))
                        epoch = state['epoch']
                        assert epoch >= num_epoch - 1

                        # Copy any checkpoints to the base folder.
                        ckpts = os.listdir(ckptspath)
                        ckpts.remove('last.ckpt')
                        for ckpt in ckpts:
                            ckptpath = os.path.join(ckptspath, ckpt)
                            newpath = os.path.join(ckptspath_base, ckpt)
                            if dry_run:
                                print(f'{ckptpath}=>{newpath}')
                            else:
                                print(f'{ckptpath}=>{newpath}')
                                os.rename(ckptpath, newpath)

                        # Copy 'last.ckpt' to base folder if we're at largest number of epochs.
                        if num_epoch == num_epochs[-1]:
                            if dry_run:
                                print(f'{lastpath}=>{lastpath_base}')
                            else:
                                print(f'{lastpath}=>{lastpath_base}')
                                os.rename(lastpath, lastpath_base)
