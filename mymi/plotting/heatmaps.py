from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import *

def plot_heatmap(
    id: str,
    heatmap: np.ndarray,
    spacing: Spacing3D,
    alpha_heatmap: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    crop_margin_mm: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = 12,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    pred_data: Optional[Dict[str, np.ndarray]] = None,
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_colorbar: bool = True,
    show_legend: bool = True,
    idx: Optional[int] = None,
    view: Axis = 0, 
    **kwargs) -> None:

    # Create plot figure/axis.
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
    else:
        show = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Plot patient regions.
    size = heatmap.shape
    plot_patients(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, crop=crop, crop_margin_mm=crop_margin_mm, ct_data=ct_data, latex=latex, legend_loc=legend_loc, region_data=region_data, show=False, show_legend=False, idx=idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin_mm, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin_mm, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Box2D' type.

    # Get aspect ratio.
    if not aspect:
        aspect = get_aspect(view, spacing) 

    # Get slice data.
    heatmap_slice, _ = get_view_slice(heatmap, idx, view)

    # Crop the image.
    if crop is not None:
        heatmap_slice = crop_fn(heatmap_slice, transpose_box(crop), use_patient_coords=False)

    # Plot heatmap
    image = ax.imshow(heatmap_slice, alpha=alpha_heatmap, aspect=aspect, origin=get_origin(view))
    if show_colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)
        plt.colorbar(image, cax=cax)

    # Plot predictions.
    if pred_data is not None:
        for pred_label, pred in pred_data.items():
            if pred.sum() != 0:
                # Get slice data.
                pred_slice, _ = get_view_slice(pred, idx, view)

                # Crop the image.
                if crop:
                    pred_slice = crop_fn(pred_slice, transpose_box(crop), use_patient_coords=False)

                # Plot prediction.
                if pred_slice.sum() != 0: 
                    cmap = ListedColormap(((1, 1, 1, 0), colour))
                    ax.imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=get_origin(view))
                    ax.plot(0, 0, c=colour, label=pred_label)
                    ax.contour(pred_slice, colors=[colour], levels=[.5], linestyles='solid')

            # Plot prediction extent.
            if pred.sum() != 0 and show_pred_extent:
                # Get prediction extent.
                pred_extent = extent(pred)

                # Plot if extent box is in view.
                label = f'{model_name} extent' if box_intersects_view_plane(pred_extent, view, z) else f'{model_name} extent (offscreen)'
                plot_box_slice(pred_extent, view, colour=colour, crop=crop, label=label, linestyle='dashed')

    # Show legend.
    if show_legend:
        plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })