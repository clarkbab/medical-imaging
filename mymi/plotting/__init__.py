from .plotting import Plotting

plotting = Plotting()

def list_saved_figures(*args, **kwargs):
    return plotting.list_saved_figures(*args, **kwargs)

def plot_batch(*args, **kwargs):
    return plotting.plot_batch(*args, **kwargs)

def plot_patient(*args, **kwargs):
    return plotting.plot_patient(*args, **kwargs)

def plot_ct_distribution(*args, **kwargs):
    return plotting.plot_ct_distribution(*args, **kwargs)

def plot_saved_figure(*args, **kwargs):
    return plotting.plot_saved_figure(*args, **kwargs)
