import os

from mymi.training import finetune_unigradicon

dataset = 'PMCC-REIRRAD-CP'
steps = int(1e5)
run_name = f'finetune_spl_steps:{steps}'
# Must specify the run name here, as we can't pass shell input during slurm job.
# We've overwritten the model weights folder within 'icon' to use our 'models' folder.
os.environ['FOOTSTEPS_NAME'] = run_name
finetune_unigradicon(dataset, run_name)
