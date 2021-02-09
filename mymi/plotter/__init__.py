from .plotter import Plotter

def plot_batch(*args, **kwargs):
    return Plotter.plot_batch(*args, **kwargs)

def plot_ct(*args, **kwargs):
    return Plotter.plot_ct(*args, **kwargs)
