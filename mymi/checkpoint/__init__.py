from .checkpoint import Checkpoint

checkpoint = Checkpoint

def load(name):
    return checkpoint.load(name)

def save(model, optimiser):
    checkpoint.save(model, optimiser)
