from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from mymi import cache
from mymi import dataset
from mymi.dataset import PatientDataExtractor

class Plotter:
    @classmethod
    def plot_ct(cls, pat_id, slice_idx, axes=True, figsize=(8, 8), labels=True, plane='axial', regions=None, transforms=[]):
        """
        effect: plots a CT slice with contours.
        args:
            pat_id: the patient ID.
            slice_idx: the slice to plot.
        kwargs:
            axes: display the pixel values on the axes.
            figsize: the size of the plot in inches.
            labels: display contours.
            plane: the viewing plane.
            regions: the regions-of-interest to plot.
            transforms: the transforms to apply before plotting.
        """
        # Load CT data.
        pat_ext = PatientDataExtractor(pat_id)
        ct_data = pat_ext.get_data()

        # Load labels.
        if labels:
            label_data = pat_ext.get_labels()

        # Find slice in correct plane.
        data_index = [
            slice_idx if plane == 'sagittal' else slice(ct_data.shape[0]),
            slice_idx if plane == 'coronal' else slice(ct_data.shape[1]),
            slice_idx if plane == 'axial' else slice(ct_data.shape[2]),
        ]
        ct_slice_data = ct_data[data_index]

        # Convert from our co-ordinate system (frontal, sagittal, longitudinal) to 
        # that required by 'imshow'.
        ct_slice_data = cls.to_image_coords(ct_slice_data, plane)

        # Determine aspect ratio.
        summary = dataset.patient_summary(pat_id).iloc[0].to_dict()
        if plane == 'axial':
            aspect = summary['spacing-y'] / summary['spacing-x']
        elif plane == 'coronal':
            aspect = summary['spacing-z'] / summary['spacing-x']
        elif plane == 'sagittal':
            aspect = summary['spacing-z'] / summary['spacing-y']

        # Plot CT data.
        plt.figure(figsize=figsize)
        plt.imshow(ct_slice_data, cmap='gray', aspect=aspect)

        if labels:
            # Plot labels.
            if len(label_data) != 0:
                # Define color pallete.
                palette = plt.cm.tab20

                # Plot each label.
                for i, (lname, ldata) in enumerate(label_data):
                    # Convert data to 'imshow' co-ordinate system.
                    ldata = ldata[data_index]
                    ldata = cls.to_image_coords(ldata, plane)

                    # Skip label if not present on this slice.
                    if ldata.max() == 0:
                        continue
                    
                    # Create binary colormap for each label.
                    colours = [(1.0, 1.0, 1.0, 0), palette(i)]
                    label_cmap = ListedColormap(colours)

                    # Plot label.
                    plt.imshow(ldata, cmap=label_cmap, aspect=aspect, alpha=0.5)
                    plt.plot(0, 0, c=palette(i), label=lname)

                # Turn on legend.
                legend = plt.legend(loc=(1.05, 0.8))
                for l in legend.get_lines():
                    l.set_linewidth(8)

        # Show axis markers.
        axes = 'on' if axes else 'off'
        plt.axis(axes)

        # Determine number of slices.
        if plane == 'axial':
            num_slices = summary['res-z']
        elif plane == 'coronal':
            num_slices = summary['res-y']
        elif plane == 'sagittal':
            num_slices = summary['res-x'] 

        # Add title.
        plt.title(f"{plane.capitalize()} slice: {slice_idx}/{num_slices - 1}")

        plt.show()

    @classmethod
    def to_image_coords(cls, data, plane):
        if plane == 'axial':
            data = np.transpose(data)
        elif plane == 'coronal' or plane == 'sagittal':
            data = np.rot90(np.transpose(data), k=2)

        return data