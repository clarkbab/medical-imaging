from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import ndarray
import os
import sys
import torch
from torchio import LabelMap, ScalarImage, Subject
from typing import *

from mymi import cache
from mymi import config
from mymi import dataset

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
        alpha: float = 0.5,
        aspect: float = 1,
        clear_cache: bool = False,
        figsize: Tuple[int, int] = (8, 8),
        labels: Union[str, Sequence[str]] = 'all',
        perimeter_only: bool = False,
        show_axes: bool = True,
        transform: str = None,
        view: Union['axial', 'coronal', 'sagittal'] = 'axial',
        window: Tuple[float, float] = None) -> None:
        """
        effect: plots a CT slice with labels.
        args:
            id: the patient ID.
            slice_idx: the slice to plot.
        kwargs:
            alpha: the label alpha.
            aspect: use a hard-coded aspect ratio, useful for viewing transformed images.
            clear_cache: force cache to clear.
            figsize: the size of the plot in inches.
            labels: the labels to display.
            perimeter_only: show the label perimeter only.
            show_axes: display the pixel values on the axes.
            transform: apply the transform before plotting.
            view: the viewing axis.
            window: the HU window to apply.
        """
        # Load patient summary.
        summary = dataset.patient(id).ct_summary(clear_cache=clear_cache).iloc[0].to_dict()

        # Load CT data.
        ct_data = dataset.patient(id).ct_data(clear_cache=clear_cache)

        # Load labels.
        if labels:
            label_data = dataset.patient(id).label_data(clear_cache=clear_cache, labels=labels)

        # Transform data.
        if transform:
            # Add 'batch' dimension.
            ct_data = np.expand_dims(ct_data, axis=0)
            label_data = dict(((n, np.expand_dims(d, axis=0)) for n, d in label_data.items()))

            # Create 'subject'.
            affine = np.array([
                [summary['spacing-x'], 0, 0, 0],
                [0, summary['spacing-y'], 0, 0],
                [0, 0, summary['spacing-z'], 0],
                [0, 0, 0, 1]
            ])
            ct_data = ScalarImage(tensor=ct_data, affine=affine)
            label_data = dict(((n, LabelMap(tensor=d, affine=affine)) for n, d in label_data.items()))

            # Transform CT data.
            subject = Subject(one_image=ct_data)
            output = transform(subject)

            # Transform label data.
            det_transform = output.get_composed_history()
            label_data = dict(((n, det_transform(Subject(a_segmentation=d))) for n, d in label_data.items()))

            # Extract results.
            ct_data = output['one_image'].data.squeeze(0)
            label_data = dict(((n, out['a_segmentation'].data.squeeze(0)) for n, d in label_data.items()))

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
        ct_slice_data = self.to_image_coords(ct_slice_data, view)

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

        if labels:
            # Plot labels.
            if len(label_data) != 0:
                # Define color pallete.
                palette = plt.cm.tab20

                # Plot each label.
                show_legend = False
                for i, (lname, ldata) in enumerate(label_data.items()):
                    # Convert data to 'imshow' co-ordinate system.
                    ldata = ldata[data_index]
                    ldata = self.to_image_coords(ldata, view)

                    # Skip label if not present on this slice.
                    if ldata.max() == 0:
                        continue
                    show_legend = True

                    # Get binary perimeter.
                    if perimeter_only:
                        ldata = self.binary_perimeter(ldata)
                    
                    # Create binary colormap for each label.
                    colours = [(1.0, 1.0, 1.0, 0), palette(i)]
                    label_cmap = ListedColormap(colours)

                    # Plot label.
                    plt.imshow(ldata, cmap=label_cmap, aspect=aspect, alpha=alpha)
                    plt.plot(0, 0, c=palette(i), label=lname)

                # Turn on legend.
                if show_legend: 
                    legend = plt.legend(loc=(1.05, 0.8))
                    for l in legend.get_lines():
                        l.set_linewidth(8)

        # Show axis markers.
        axes = 'on' if show_axes else 'off'
        plt.axis(axes)

        # Determine number of slices.
        if view == 'axial':
            num_slices = ct_data.shape[2]
        elif view == 'coronal':
            num_slices = ct_data.shape[1]
        elif view == 'sagittal':
            num_slices = ct_data.shape[0]

        # Add title.
        plt.title(f"{view.capitalize()} slice: {slice_idx}/{num_slices - 1}")

        plt.show()

    def binary_perimeter(
        self,
        label: ndarray) -> ndarray:
        """
        returns: label containing perimeter pixels only.
        args:
            label: the full label.
        """
        label_perimeter = np.zeros_like(label, dtype=bool)
        x_dim, y_dim = label.shape
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (label[i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and label[i - 1, j] == 0 or 
                    i != x_dim - 1 and label[i + 1, j] == 0 or
                    j != 0 and label[i, j - 1] == 0 or
                    j != y_dim - 1 and label[i, j + 1] == 0)):
                    label_perimeter[i, j] = 1

        return label_perimeter

    def to_image_coords(
        self,
        data: ndarray,
        view: Union['axial', 'coronal', 'sagittal']) -> ndarray:
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
