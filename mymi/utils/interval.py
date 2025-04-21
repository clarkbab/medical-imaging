from mymi.typing import *

def interval_matches(
    epoch: int,
    step: int,
    interval: TrainingInterval,
    n_steps_per_epoch) -> bool:
    # Parse interval.
    terms = interval.split(':')
    n_terms = len(terms)
    assert n_terms in (2, 4)

    if len(terms) == 2:
        if terms[0] == 'step':
            interval_step = int(terms[1])
            return step % interval_step == 0
        elif terms[0] == 'epoch':
            interval_epoch = int(terms[1]) if terms[1].isdigit() else terms[1]
            return epoch_matches(epoch, step, interval_epoch, n_steps_per_epoch)
    elif len(terms) == 4:
        interval_epoch = int(terms[1]) if terms[1].isdigit() else terms[1]
        return epoch_matches(epoch, step, interval_epoch, n_steps_per_epoch) and step % interval_step == 0

def epoch_matches(
    epoch: int,
    step: int,
    interval_epoch: Union[int, Literal['start', 'end']],
    n_steps_per_epoch: int) -> bool:
    if isinstance(interval_epoch, int):
        return epoch % interval_epoch == 0
    elif interval_epoch == 'start':
        return epoch % n_steps_per_epoch == 0
    elif interval_epoch == 'end':
        return step % n_steps_per_epoch == n_steps_per_epoch - 1
