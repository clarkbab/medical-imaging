from .plotter import Plotter

def list_saved_figures(*args, **kwargs):
    return Plotter.list_saved_figures(*args, **kwargs)

def plot_batch(*args, **kwargs):
    return Plotter.plot_batch(*args, **kwargs)

def plot_patient(*args, **kwargs):
    return Plotter.plot_patient(*args, **kwargs)

def plot_ct_distribution(*args, **kwargs):
    return Plotter.plot_ct_distribution(*args, **kwargs)

def plot_saved_figure(*args, **kwargs):
    return Plotter.plot_saved_figure(*args, **kwargs)
