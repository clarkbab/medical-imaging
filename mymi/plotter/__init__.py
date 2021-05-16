from .plotter import Plotter

plotter = Plotter()

def list_saved_figures(*args, **kwargs):
    return plotter.list_saved_figures(*args, **kwargs)

def plot_batch(*args, **kwargs):
    return plotter.plot_batch(*args, **kwargs)

def plot_patient(*args, **kwargs):
    return plotter.plot_patient(*args, **kwargs)

def plot_ct_distribution(*args, **kwargs):
    return plotter.plot_ct_distribution(*args, **kwargs)

def plot_saved_figure(*args, **kwargs):
    return plotter.plot_saved_figure(*args, **kwargs)
