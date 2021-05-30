from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import re
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Literal, Sequence, Tuple, Union

from mymi import dataset
from mymi.regions import is_region, RegionColours

class Plotter:
    def plot_ct_distribution(
        self, 
        bin_width: int = 10,
        clear_cache: bool = False,
        figsize: Tuple[int, int] = (10, 10),
        labels: Union[str, Sequence[str]] = 'all',
        max_bin: int = None,
        min_bin: int = None,
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> None:
        """
        effect: plots CT distribution of the dataset.
        kwargs:
            bin_width: the width of the histogram bins.
            clear_cache: forces the cache to clear.
            figsize: the size of the figure.
            labels: include patients with any of the listed labels (behaves like an OR).
            max_bin: the maximum bin to show. 
            min_bin: the minimum bin to show.
            num_pats: the number of patients to include.
            pat_ids: the patients to include.
        """
        # Load CT distribution.
        freqs = dataset.ct_distribution(bin_width=bin_width, clear_cache=clear_cache, labels=labels, num_pats=num_pats, pat_ids=pat_ids)

        # Remove bins we don't want.
        if min_bin or max_bin:
            for b in freqs.keys():
                if (min_bin and b < min_bin) or (max_bin and b > max_bin):
                    freqs.pop(b)

        # Plot the histogram.
        plt.figure(figsize=figsize)
        keys = tuple(freqs.keys())
        values = tuple(freqs.values())
        plt.hist(keys[:-1], keys, weights=values[:-1])
        plt.show()

    def plot_patient(
        self,
        id: str,
        slice_idx: int,
        alpha: float = 0.2,
        aspect: float = 1,
        axes: bool = True,
        clear_cache: bool = False,
        figsize: Tuple[int, int] = (8, 8),
        font_size: int = 10,
        internal_regions: bool = False,
        latex: bool = False,
        legend: bool = True,
        legend_loc: Union[str, Tuple[float, float]] = 'upper right',
        legend_size: int = 10,
        perimeter: bool = True,
        regions: Union[str, Sequence[str]] = 'all',
        title: Union[bool, str] = True,
        transform: torchio.transforms.Transform = None,
        view: Literal['axial', 'coronal', 'sagittal'] = 'axial',
        window: Tuple[float, float] = None) -> None:
        """
        effect: plots a CT slice with labels.
        args:
            id: the patient ID.
            slice_idx: the slice to plot.
        kwargs:
            alpha: the region alpha.
            aspect: use a hard-coded aspect ratio, useful for viewing transformed images.
            axes: display the axes ticks and labels.
            clear_cache: force cache to clear.
            figsize: the size of the plot in inches.
            font_size: the size of the font.
            internal_regions: use the internal MYMI region names.
            latex: use latex to display text.
            legend: display the legend.
            legend_loc: the location of the legend.
            perimeter: highlight the perimeter.
            regions: the regions to display.
            title: turns the title on/off. Can optionally pass a custom title.
            transform: apply the transform before plotting.
            view: the viewing axis.
            window: the HU window to apply.
        """
        # Get params.
        plt.rcParams.update({
            'font.size': font_size
        })

        # Set latex params.
        if latex:
            plt.rcParams.update({
                "font.family": "serif",
                'text.usetex': True
            })

        # Load patient summary.
        summary = dataset.patient(id).ct_summary(clear_cache=clear_cache).iloc[0].to_dict()

        # Load CT data.
        ct_data = dataset.patient(id).ct_data(clear_cache=clear_cache)

        if regions:
            # Load region data.
            region_data = dataset.patient(id).region_data(clear_cache=clear_cache, regions=regions)

            if internal_regions:
                # Map to internal region names.
                region_data = dict((self._to_internal_region(r, clear_cache=clear_cache), d) for r, d in region_data.items())

        # Transform data.
        if transform:
            # Add 'batch' dimension.
            ct_data = np.expand_dims(ct_data, axis=0)
            region_data = dict(((n, np.expand_dims(d, axis=0)) for n, d in region_data.items()))

            # Create 'subject'.
            affine = np.array([
                [summary['spacing-x'], 0, 0, 0],
                [0, summary['spacing-y'], 0, 0],
                [0, 0, summary['spacing-z'], 0],
                [0, 0, 0, 1]
            ])
            ct_data = ScalarImage(tensor=ct_data, affine=affine)
            region_data = dict(((n, LabelMap(tensor=d, affine=affine)) for n, d in region_data.items()))

            # Transform CT data.
            subject = Subject(input=ct_data)
            output = transform(subject)

            # Transform region data.
            det_transform = output.get_composed_history()
            region_data = dict(((r, det_transform(Subject(region=d))) for r, d in region_data.items()))

            # Extract results.
            ct_data = output['input'].data.squeeze(0)
            region_data = dict(((n, o['region'].data.squeeze(0)) for n, o in region_data.items()))

        # Find slice in correct plane, x=sagittal, y=coronal, z=axial.
        assert view in ('axial', 'coronal', 'sagittal')
        data_index = (
            slice_idx if view == 'sagittal' else slice(ct_data.shape[0]),
            slice_idx if view == 'coronal' else slice(ct_data.shape[1]),
            slice_idx if view == 'axial' else slice(ct_data.shape[2]),
        )
        ct_slice_data = ct_data[data_index]

        # Convert from our co-ordinate system (frontal, sagittal, longitudinal) to 
        # that required by 'imshow'.
        ct_slice_data = self._to_image_coords(ct_slice_data, view)

        # Only apply aspect ratio if no transforms are being presented otherwise
        # we might end up with skewed images.
        if not transform:
            if view == 'axial':
                aspect = summary['spacing-y'] / summary['spacing-x']
            elif view == 'coronal':
                aspect = summary['spacing-z'] / summary['spacing-x']
            elif view == 'sagittal':
                aspect = summary['spacing-z'] / summary['spacing-y']

        # Determine plotting window.
        if window:
            vmin, vmax = window
        else:
            vmin, vmax = ct_data.min(), ct_data.max()

        # Plot CT data.
        plt.figure(figsize=figsize)
        plt.imshow(ct_slice_data, cmap='gray', aspect=aspect, vmin=vmin, vmax=vmax)

        # Add axis labels.
        if axes:
            plt.xlabel('voxel')
            plt.ylabel('voxel')

        if regions:
            # Plot regions.
            if len(region_data) != 0:
                # Create palette if not using internal region colours.
                if not internal_regions:
                    palette = plt.cm.tab20

                # Plot each region.
                show_legend = False     # Only show legend if slice has at least one region.
                for i, (region, data) in enumerate(region_data.items()):
                    # Convert data to 'imshow' co-ordinate system.
                    data = data[data_index]
                    data = self._to_image_coords(data, view)

                    # Skip region if not present on this slice.
                    if data.max() == 0:
                        continue
                    else:
                        show_legend = True
                    
                    # Create binary colormap for each region.
                    if internal_regions:
                        colour = getattr(RegionColours, region)
                    else:
                        colour = palette(i)
                    colours = [(1.0, 1.0, 1.0, 0), colour]
                    region_cmap = ListedColormap(colours)

                    # Plot region.
                    plt.imshow(data, cmap=region_cmap, aspect=aspect, alpha=alpha)
                    label = self._escape_latex(region) if latex else region
                    plt.plot(0, 0, c=colour, label=label)
                    if perimeter:
                        plt.contour(data, colors=[colour], levels=[0.5])

                # Turn on legend.
                if legend and show_legend: 
                    # Get legend props.
                    plt_legend = plt.legend(loc=legend_loc, prop={'size': legend_size})
                    for l in plt_legend.get_lines():
                        l.set_linewidth(8)

        # Show axis markers.
        show_axes = 'on' if axes else 'off'
        plt.axis(show_axes)

        # Determine number of slices.
        if view == 'axial':
            num_slices = ct_data.shape[2]
        elif view == 'coronal':
            num_slices = ct_data.shape[1]
        elif view == 'sagittal':
            num_slices = ct_data.shape[0]

        # Add title.
        if title:
            if isinstance(title, str):
                title_text = title
            else:
                title_text = f"{view.capitalize()} slice: {slice_idx}/{num_slices - 1}" 

            # Escape text if using latex.
            if latex:
                title_text = self._escape_latex(title_text)

            plt.title(title_text)

        plt.show()

    def binary_perimeter(
        self,
        data: ndarray) -> ndarray:
        """
        returns: a 2D array containing perimeter pixels only.
        args:
            data: the 2D binary array.
        """
        data_perimeter = np.zeros_like(data, dtype=bool)
        x_dim, y_dim = data.shape
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (data[i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and data[i - 1, j] == 0 or 
                    i != x_dim - 1 and data[i + 1, j] == 0 or
                    j != 0 and data[i, j - 1] == 0 or
                    j != y_dim - 1 and data[i, j + 1] == 0)):
                    data_perimeter[i, j] = 1

        return data_perimeter

    def _to_image_coords(
        self,
        data: ndarray,
        view: Literal['axial', 'coronal', 'sagittal']) -> ndarray:
        """
        returns: data in correct orientation for viewing.
        args:
            data: the data to orient.
            view: the viewing axis.
        """
        if view == 'axial':
            data = np.transpose(data)
        elif view in ('coronal', 'sagittal'):
            data = np.rot90(data)

        return data
    
    def _to_internal_region(
        self,
        region: str,
        clear_cache: bool = False) -> str:
        """
        returns: the internal region name.
        args:
            region: the dataset region name.
        kwargs:
            clear_cache: force the cache to clear.
        """
        # Check if region is an internal name.
        if is_region(region):
            return region

        # Map from dataset name to internal name.
        map_df = dataset.region_map(clear_cache=clear_cache)
        map_dict = dict((r.dataset, r.internal) for _, r in map_df.iterrows())
        if region in map_dict:
            return map_dict[region]

        # Raise an error if we don't know how to translate to the internal name.
        raise ValueError(f"Region '{region}' is neither an internal region, nor listed in the region map, can't create internal name.")

    def _escape_latex(
        self,
        text: str) -> str:
        """
        returns: a string with escaped latex special characters.
        args:
            text: the string to escape.
        """
        # Provide map for special characters.
        char_map = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
            '<': r'\textless{}',
            '>': r'\textgreater{}',
        }
        regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(char_map.keys(), key = lambda item: - len(item))))
        return regex.sub(lambda match: char_map[match.group()], text)
    