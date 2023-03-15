import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional

from mymi import config

def load_data(name: str) -> Dict[str, np.ndarray]:
    filepath = os.path.join(config.directories.files, 'lr-finder', f'{name}.json')
    data = json.load(open(filepath))
    return data

def suggestion(
    name: str,
    skip_end: Optional[int] = None,
    skip_start: int = 0,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> float:
    # Get loss data.
    results = load_data(name)
    losses = results['loss']
    if skip_end is not None:
        losses = losses[skip_start:-skip_end]
    else:
        losses = losses[skip_start:]

    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)

    # Get minimum gradient.
    min_grad_idx = np.gradient(losses).argmin()
    min_grad_idx = skip_start + min_grad_idx

    # Get associated learning rate.
    lr = results['lr'][min_grad_idx]

    return lr

def plot(
    name: str,
    smooth: bool = False,
    smooth_kernel_sd: float = 1) -> None:
    # Load learning rates and losses.
    results = load_data(name)
    lr = results['lr']
    losses = results['loss']
    
    # Smooth the signal.
    if smooth:
        losses = gaussian_filter1d(losses, smooth_kernel_sd)
        losses = apply_smoothing(losses)

    # Plot.
    plt.plot(lr, losses)
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.xscale('log')
    plt.show()

# From: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def apply_smoothing(x: List[float],window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if len(x) < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    y = y[:len(x)]

    return y
