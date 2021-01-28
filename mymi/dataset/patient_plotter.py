from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import cache
from mymi import dataset
from mymi.dataset.dicom import PatientDataExtractor

class PatientPlotter:
    def __init__(self, pat_id):
        """
        pat_id: a patient ID string.
        """
        self.pat_id = pat_id

    def plot_ct(self, slice_idx, axis='on', figsize=(8, 8), plane='axial', regions=None, transforms=[]):
        """
        effect: plots a CT slice with contours.
        figsize: the size of the plot in inches.
        plane: the viewing plane.
        regions: the regions-of-interest to plot.
        """
        # Load CT data and labels.
        pat_ext = PatientDataExtractor(self.pat_id)
        ct_data = pat_ext.get_data()

        # Load labels.
        labels = pat_ext.get_labels()

        # Plot CT slice.
        data_index = [
            slice_idx if plane == 'sagittal' else slice(ct_data.shape[0]),
            slice_idx if plane == 'coronal' else slice(ct_data.shape[1]),
            slice_idx if plane == 'axial' else slice(ct_data.shape[2]),
        ]
        ct_slice_data = ct_data[data_index]
        plt.figure(figsize=figsize)
        # TODO: Handle pixel aspect.
        plt.imshow(np.transpose(ct_slice_data), cmap='gray')

        # Plot labels.
        if len(labels) != 0:
            colour_gen = plt.cm.tab10

            # Plot each label.
            for i, (label_name, label_data) in enumerate(labels):
                label_data = label_data[data_index]
                colours = [(1.0, 1.0, 1.0, 0), colour_gen(i)]
                label_cmap = ListedColormap(colours)
                plt.imshow(np.transpose(label_data), cmap=label_cmap, alpha=0.5)
                plt.plot(0, 0, c=colour_gen(i), label=label_name)

            # Turn on legend.
            plt.legend(loc=(1.05, 0.8))

        plt.axis(axis)
        plt.show()