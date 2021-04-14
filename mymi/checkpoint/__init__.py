from .checkpoint import Checkpoint

checkpoint = Checkpoint

def load(*args, **kwargs):
    return checkpoint.load(*args, **kwargs)

def save(*args, **kwargs):
    checkpoint.save(*args, **kwargs)
