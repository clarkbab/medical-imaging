import imageio
import os
import PIL
import tempfile
from IPython.display import Image as IPythonImage, display
from tqdm import tqdm
from typing import *

from mymi.typing import *


def make_gif(
    plot_fn: Callable[..., PIL.Image.Image],
    plot_args: List[Dict[str, Any]],
    frame_time: float = 0.5,
    loop: bool = True,
    n_frames: int | None = None,
    overwrite: bool = False,
    pause_time: float = 5,
    savepath: FilePath | None = None,
    width: int = 1000,
    ) -> None:
    if savepath is not None and os.path.exists(savepath) and not overwrite:
        with open(savepath, 'rb') as f:
            display(IPythonImage(data=f.read(), format='png', width=width))
        return

    if savepath is None:
        savepath = tempfile.NamedTemporaryFile(suffix='.apng', delete=False).name

    # Generate frames by calling the plotting function with each set of args.
    png_images = []
    for i, kwargs in tqdm(enumerate(plot_args), total=len(plot_args), desc="Creating GIF frames"):
        png_image = plot_fn(**kwargs)
        png_images.append(png_image)
        if n_frames is not None and i + 1 >= n_frames:
            break

    # Save animated PNG (avoids GIF's 256-colour palette limitation).
    frames = png_images
    frames_per_second = 1 / frame_time
    frames = frames + [frames[-1]] * int(pause_time / frame_time)
    imageio.mimsave(savepath, frames, fps=frames_per_second, loop=0 if loop else None)

    with open(savepath, 'rb') as f:
        display(IPythonImage(data=f.read(), format='png', width=width))
