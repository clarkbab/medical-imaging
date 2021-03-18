from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from torchio import LabelMap, ScalarImage, Subject

from mymi import cache
from mymi import dataset

class Plotter:
    @classmethod
    def plot_acquisition_params_boxplot(cls):
        pass

    @classmethod
    def plot_distribution(cls, region='all'):
        """
        effect: plots the distribution of data in the dataset.
        kwargs:
            region: only include patients who have a particular region/s.
        """
        pass

    @classmethod
    def plot_patient(cls, pat_id, slice_idx, aspect=None, axes=True, figsize=(8, 8), full_label=False, labels=True, view='axial', regions='all', transform=None, window=None):
        """
        effect: plots a CT slice with contours.
        args:
            pat_id: the patient ID.
            slice_idx: the slice to plot.
        kwargs:
            aspect: use a hard-coded aspect ratio, useful for viewing transformed images.
            axes: display the pixel values on the axes.
            figsize: the size of the plot in inches.
            full_label: should we should full label mask or perimeter?
            view: the viewing plane.
                axial: viewed from the cranium (slice_idx=0) to the caudal region.
                coronal: viewed from the anterior (slice_idx=0) to the posterior.
                sagittal: viewed from the 
            regions: the regions-of-interest to plot.
            window: the HU window to apply.
        """
        # Load patient summary.
        summary = dataset.patient_summary(pat_id)

        # Load CT data.
        ct_data = dataset.patient_data(pat_id)

        # Load labels.
        if regions is not None:
            label_data = dataset.patient_labels(pat_id, regions=regions)

        # Transform data.
        if transform:
            # Add 'batch' dimension.
            ct_data = np.expand_dims(ct_data, axis=0)
            label_data = [(name, np.expand_dims(data, axis=0)) for name, data in label_data]

            # Create 'subject'.
            affine = np.array([
                [summary['spacing-x'].item(), 0, 0, 0],
                [0, summary['spacing-y'].item(), 0, 0],
                [0, 0, summary['spacing-z'].item(), 0],
                [0, 0, 0, 1]
            ])
            ct_data = ScalarImage(tensor=ct_data, affine=affine)
            label_data = [(name, LabelMap(tensor=data, affine=affine)) for name, data in label_data]

            # Transform CT data.
            subject = Subject(one_image=ct_data)
            output = transform(subject)

            # Transform label data.
            det_transform = output.get_composed_history()
            label_data = [(name, det_transform(Subject(a_segmentation=image))) for name, image in label_data]

            # Extract results.
            ct_data = output['one_image'].data.squeeze(0)
            label_data = [(name, out['a_segmentation'].data.squeeze(0)) for name, out in label_data]

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
        ct_slice_data = cls.to_image_coords(ct_slice_data, view)

        # Only apply aspect ratio if no transforms are being presented otherwise
        # we might end up with skewed images.
        if not transform:
            if view == 'axial':
                aspect = summary['spacing-y'] / summary['spacing-x']
            elif view == 'coronal':
                aspect = summary['spacing-z'] / summary['spacing-x']
            elif view == 'sagittal':
                aspect = summary['spacing-z'] / summary['spacing-y']
        else:
            aspect = 1

        # Determine plotting window.
        if window is not None:
            vmin, vmax = window
        else:
            vmin, vmax = ct_data.min(), ct_data.max()

        # Plot CT data.
        plt.figure(figsize=figsize)
        plt.imshow(ct_slice_data, cmap='gray', aspect=aspect, vmin=vmin, vmax=vmax)

        if regions is not None:
            # Plot labels.
            if len(label_data) != 0:
                # Define color pallete.
                palette = plt.cm.tab20

                # Plot each label.
                show_legend = False
                for i, (lname, ldata) in enumerate(label_data):
                    # Convert data to 'imshow' co-ordinate system.
                    ldata = ldata[data_index]
                    ldata = cls.to_image_coords(ldata, view)

                    # Skip label if not present on this slice.
                    if ldata.max() == 0:
                        continue
                    show_legend = True

                    # Get binary perimeter.
                    if not full_label:
                        ldata = cls.binary_perimeter(ldata)
                    
                    # Create binary colormap for each label.
                    colours = [(1.0, 1.0, 1.0, 0), palette(i)]
                    label_cmap = ListedColormap(colours)

                    # Plot label.
                    plt.imshow(ldata, cmap=label_cmap, aspect=aspect, alpha=0.5)
                    plt.plot(0, 0, c=palette(i), label=lname)

                # Turn on legend.
                if show_legend: 
                    legend = plt.legend(loc=(1.05, 0.8))
                    for l in legend.get_lines():
                        l.set_linewidth(8)

        # Show axis markers.
        axes = 'on' if axes else 'off'
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

    @classmethod
    def to_image_coords(cls, data, view):
        if view == 'axial':
            data = np.transpose(data)
        elif view in ('coronal', 'sagittal'):
            data = np.rot90(data)

        return data

    @classmethod
    def plot_batch(cls, input, slice_idx, label=None, pred=None, aspect=1.0, figsize=(8, 8), full_label=False, num_images='all', view='axial', return_figure=False):
        """
        effect: plots a training batch.
        args:
            input: the input data.
            slice_idx: the slice to show. Can be list-like if batch has more than one element.
        kwargs:
            figsize: the plot figure size.
            full_label: should we should full label mask or perimeter?
            label: the label data.
            num_images: number of images from the batch to plot.
            pred: the predicted segmentation mask.
            view: the viewing plane.
            return_figure: whether we return or plot the figure.
        """
        # Convert to CPU.
        input = input.cpu()
        if label is not None: label = label.cpu()
        if pred is not None: pred = pred.cpu()

        # Remove input 'channel' dimension.
        if input.dim() == 5:
            input = input.squeeze(1)

        # Convert output 'channel' to prediction.
        if pred.dim() == 5:
            pred = pred.argmax(1)  

        # Get input data.
        assert view in ('axial', 'coronal', 'sagittal')
        num_images = len(input) if num_images == 'all' else num_images
        input_data = input[:num_images]

        # Handle 'slice_idx'.
        if isinstance(slice_idx, int):
            slice_idx = [slice_idx] * num_images

        # Add label.
        if label is not None:
            # Get subset of images.
            label_data = label[:num_images]

        # Add prediction.
        if pred is not None:
            # Get subset of images.
            pred_data = pred[:num_images]

        # Plot data.
        fig, axs = plt.subplots(num_images, figsize=figsize)
        if num_images == 1: axs = [axs]
        if return_figure: plt.close(fig)

        for i in range(num_images):
            # Plot input slice data.
            if view == 'axial':
                slice_input_data = np.transpose(input_data[i, :, :, slice_idx[i]])
            elif view == 'coronal':
                slice_input_data = np.rot90(input_data[i, :, slice_idx[i], :])
            elif view == 'sagittal':
                slice_input_data = np.rot90(input_data[i, slice_idx[i], :, :])

            axs[i].imshow(slice_input_data, aspect=aspect, cmap='gray')

            # Plot prediction slice data.
            if pred is not None:
                if view == 'axial':
                    slice_pred_data = np.transpose(pred_data[i, :, :, slice_idx[i]])
                elif view == 'coronal':
                    slice_pred_data = np.rot90(pred_data[i, :, slice_idx[i], :])
                elif view == 'sagittal':
                    slice_pred_data = np.rot90(pred_data[i, slice_idx[i], :, :])

                # Create binary colormap.
                colours = [(1.0, 1.0, 1.0, 0), (0.12, 0.47, 0.7, 1.0)]
                label_cmap = ListedColormap(colours)

                axs[i].imshow(slice_pred_data, aspect=aspect, cmap=label_cmap)

            # Plot label slice data.
            if label is not None:
                if view == 'axial':
                    slice_label_data = np.transpose(label_data[i, :, :, slice_idx[i]])
                elif view == 'coronal':
                    slice_label_data = np.rot90(label_data[i, :, slice_idx[i], :])
                elif view == 'sagittal':
                    slice_label_data = np.rot90(label_data[i, slice_idx[i], :, :])

                # Get binary perimeter.
                if not full_label:
                    slice_label_data = cls.binary_perimeter(slice_label_data)

                # Create binary colormap.
                colours = [(1.0, 1.0, 1.0, 0), (1.0, 1.0, 1.0e-5, 1.0)]
                label_cmap = ListedColormap(colours)

                axs[i].imshow(slice_label_data, aspect=aspect, cmap=label_cmap)
        
        # Return or plot the figure.
        if return_figure:
            return fig
        else:
            plt.show()

    @classmethod
    def binary_perimeter(cls, mask):
        """
        returns: the edge pixels of the mask.
        args:
            mask: a 2D image mask.
        """
        mask_perimeter = np.zeros_like(mask, dtype=bool)
        x_dim, y_dim = mask.shape
        for i in range(x_dim):
            for j in range(y_dim):
                # Check if edge pixel.
                if (mask[i, j] == 1 and 
                    ((i == 0 or i == x_dim - 1) or
                    (j == 0 or j == y_dim - 1) or
                    i != 0 and mask[i - 1, j] == 0 or 
                    i != x_dim - 1 and mask[i + 1, j] == 0 or
                    j != 0 and mask[i, j - 1] == 0 or
                    j != y_dim - 1 and mask[i, j + 1] == 0)):
                    mask_perimeter[i, j] = 1

        return mask_perimeter
