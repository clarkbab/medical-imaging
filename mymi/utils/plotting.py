import imageio
import io
import os
import PIL
import tempfile
import threading
from IPython.display import Image as IPythonImage, display
from tqdm import tqdm
from typing import *

from mymi.typing import *

def plot_gif(
    plot_fn: Callable[..., PIL.Image.Image] | None = None,
    plot_args: List[Tuple[Any, ...]] | None = None,
    plot_kwargs: List[Dict[str, Any]] | None = None,
    frame_time: float = 0.5,
    loop: bool = True,
    n_frames: int | None = None,
    overwrite: bool = False,
    pause_time: float = 5,
    player: bool = True,
    savepath: FilePath | None = None,
    width: int = 1000,
    ) -> None:
    def _to_png_bytes(frame: Any) -> bytes:
        img = frame if isinstance(frame, PIL.Image.Image) else PIL.Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _display_player(frame_bytes: List[bytes]) -> None:
        try:
            import ipywidgets as widgets
        except ImportError as e:
            raise ImportError("'player=True' requires ipywidgets in JupyterLab.") from e

        n_local_frames = len(frame_bytes)
        image_widget = widgets.Image(value=frame_bytes[0], format='png', width=width)
        play_toggle = widgets.ToggleButton(
            value=False,
            description='▶',
            tooltip='Play / Pause',
            layout=widgets.Layout(width='48px'),
        )
        frame_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=n_local_frames - 1,
            step=1,
            description='Frame',
            continuous_update=True,
        )
        reverse_toggle = widgets.ToggleButton(
            value=False,
            description='Reverse',
            tooltip='Play frames in reverse order',
        )

        _timer = [None]

        def _display_idx(i: int) -> int:
            return (n_local_frames - 1 - i) if reverse_toggle.value else i

        def _advance():
            if not play_toggle.value:
                return
            step = -1 if reverse_toggle.value else 1
            next_val = frame_slider.value + step
            if next_val > frame_slider.max:
                next_val = frame_slider.min if loop else frame_slider.max
            elif next_val < frame_slider.min:
                next_val = frame_slider.max if loop else frame_slider.min
            frame_slider.value = next_val
            _timer[0] = threading.Timer(frame_time, _advance)
            _timer[0].start()

        def _on_play_toggle(change):
            if change['new']:
                play_toggle.description = '⏸'
                _advance()
            else:
                play_toggle.description = '▶'
                if _timer[0] is not None:
                    _timer[0].cancel()

        def _on_frame_change(change):
            image_widget.value = frame_bytes[change['new']]

        def _on_reverse(_change):
            image_widget.value = frame_bytes[frame_slider.value]

        play_toggle.observe(_on_play_toggle, names='value')
        frame_slider.observe(_on_frame_change, names='value')
        reverse_toggle.observe(_on_reverse, names='value')

        # Autoplay immediately when rendered.
        play_toggle.value = True

        display(widgets.VBox([
            widgets.HBox([play_toggle, reverse_toggle]),
            frame_slider,
            image_widget,
        ]))

    if savepath is not None and os.path.exists(savepath) and not overwrite:
        if player:
            cached_frames = imageio.mimread(savepath, memtest=False)
            if n_frames is not None:
                cached_frames = cached_frames[:n_frames]
            frame_bytes = [_to_png_bytes(frame) for frame in cached_frames]
            _display_player(frame_bytes)
            return
        with open(savepath, 'rb') as f:
            display(IPythonImage(data=f.read(), format='png', width=width))
        return

    if savepath is None:
        savepath = tempfile.NamedTemporaryFile(suffix='.apng', delete=False).name
    else:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

    n_frames = n_frames if n_frames is not None else (len(plot_args) if plot_args is not None else len(plot_kwargs))
    if plot_args is None:
        plot_args = [() for _ in range(n_frames)]
    else:
        plot_args = plot_args[:n_frames]
    if plot_kwargs is None:
        plot_kwargs = [{} for _ in range(n_frames)]
    else:
        plot_kwargs = plot_kwargs[:n_frames]
    assert len(plot_args) == len(plot_kwargs), f"'plot_args' (len={len(plot_args)}) and 'plot_kwargs' (len={len(plot_kwargs)}) must have the same length."

    # Generate frames by calling the plotting function with per-frame args/kwargs.
    png_images = []
    for i, (frame_args, frame_kwargs) in tqdm(enumerate(zip(plot_args, plot_kwargs)), total=n_frames, desc="Creating GIF frames"):
        png_image = plot_fn(*frame_args, **frame_kwargs)
        png_images.append(png_image)

    # Save animated PNG (avoids GIF's 256-colour palette limitation).
    frames = png_images
    frames_per_second = 1 / frame_time
    frames = frames + [frames[-1]] * int(pause_time / frame_time)
    imageio.mimsave(savepath, frames, fps=frames_per_second, loop=0 if loop else None)

    if player:
        # Build PNG bytes for each frame for fast widget updates.
        frame_bytes = [_to_png_bytes(img) for img in png_images]
        _display_player(frame_bytes)
        return

    with open(savepath, 'rb') as f:
        display(IPythonImage(data=f.read(), format='png', width=width))
