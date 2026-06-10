from mymi.typing import *

# Cases:
# - step:200, matches steps 0, 200, 400, etc.
# - epoch:50, matches every step in epoch 0, 50, 100, etc.
# - epoch:5,step:50, matches epoch=0,step=0,50,100,etc. , epoch=5,step=0,50,100,etc.
# - epoch:start, matches every step in the first epoch.
# - step:epoch_start, matches the first step in every epoch.
# - step:epoch_end, matches the last step in every epoch.
# - epoch:5,step:epoch_start, matches the first step in every epoch.
# - epoch:5,step:epoch_end, matches the last step in every epoch.

# step_match_length can be used to match more than one step. Not sure how this will
# work for step:epoch_end.

# Examples:
# - Logging training images: interval='step:200', step_match_length=5. Logs 5 images
#   once every 200 steps.
# - Logging validation images: interval='step:epoch_start'. step_match_length isn't 
#   very useful here as all validation batches exist at the same training step.

def interval_matches(
    step: TrainingStep,  # Global, keeps counting up across epochs.
    interval: TrainingInterval,
    n_steps_per_epoch: int,
    epoch_match_length: int = 1,  # Allows us to match more than one epoch.
    step_match_length: int = 1,  # Allows us to match more than one step.
    ) -> bool:
    epoch = step // n_steps_per_epoch
    # Parse interval.
    terms = interval.split(':')
    assert len(terms) in (2, 4)

    if len(terms) == 2:
        if terms[0] == 'step':
            if terms[1] == 'epoch_start':
                return step % n_steps_per_epoch < step_match_length
            if terms[1] == 'epoch_end':
                return step % n_steps_per_epoch >= n_steps_per_epoch - step_match_length
            interval_step = int(terms[1])
            return step % interval_step < step_match_length
        elif terms[0] == 'epoch':
            return epoch % int(terms[1]) < epoch_match_length
    elif len(terms) == 4:
        epoch_matches = epoch % int(terms[1]) < epoch_match_length
        if terms[3] == 'epoch_start':
            return epoch_matches and step % n_steps_per_epoch < step_match_length
        if terms[3] == 'epoch_end':
            return epoch_matches and step % n_steps_per_epoch >= n_steps_per_epoch - step_match_length
        interval_step = int(terms[3])
        return epoch_matches and step % interval_step < step_match_length

def test_interval_matches(
    interval: TrainingInterval,
    n_epochs: int,
    n_steps_per_epoch: int,
    epoch_match_length: int = 1,
    step_match_length: int = 1,
    ) -> list:
    matches = []
    for epoch in range(n_epochs):
        for local_step in range(n_steps_per_epoch):
            step = epoch * n_steps_per_epoch + local_step
            if interval_matches(step, interval, n_steps_per_epoch, epoch_match_length, step_match_length):
                matches.append((epoch, step))
    return matches
