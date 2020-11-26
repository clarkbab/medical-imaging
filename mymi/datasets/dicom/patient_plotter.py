from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientDataExtractor

class PatientPlotter:
    @staticmethod
    def from_id(pat_id, dataset=ds):
        """
        pat_id: an identifier for the patient.
        returns: PatientSummary object.
        """
        if not dataset.has_id(pat_id):
            print(f"Patient ID '{pat_id}' not found in dataset.")
            # raise error
            exit(0)

        # TODO: Read an env var or something to make dataset implicit.
        
        return PatientPlotter(pat_id, dataset=dataset)

    def __init__(self, pat_id, dataset=ds):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.dataset = dataset
        self.pat_id = pat_id

    def plot_ct(self, slice_idx, contours=None, figsize=(8, 8), plane='axial', transform=False):
        """
        effect: plots a CT slice with contours.
        contours: the contours to plot.
        figsize: the size of the plot in inches.
        plane: the viewing plane.
        """
        # Load CT data and labels.
        pat_ext = PatientDataExtractor(self.pat_id, dataset=self.dataset)
        ct_data = pat_ext.get_data(transform=transform)

        # Load labels.
        labels = pat_ext.list_labels()
        label_names = [l[0] for l in labels] 

        # Filter unwanted labels.
        def should_contour(label):
            label_name = label[0]
            return ((type(contours) == str and contours == 'all' or contours == label_name) or
                (type(contours) == list and label_name in contours))
        labels = list(filter(should_contour, labels))

        # Someone probably typed the wrong label name.
        if len(labels) == 0 and contours != None:
            print(f"No label matching '{contours}'.")
            print(f"Available contours: {label_names}.")

        # Plot data slice.
        data_index = [
            slice_idx if plane == 'sagittal' else slice(ct_data.shape[0]),
            slice_idx if plane == 'coronal' else slice(ct_data.shape[1]),
            slice_idx if plane == 'axial' else slice(ct_data.shape[2]),
        ]
        ct_slice_data = ct_data[data_index]
        plt.figure(figsize=figsize)
        # TODO: Handle pixel aspect.
        plt.imshow(np.transpose(ct_slice_data), cmap='gray')

        if len(labels) != 0:
            # Create colour generator. 
            colour_gen = plt.cm.tab10

            # Plot each label.
            for i, (label_name, label_data) in enumerate(labels):
                label_data = label_data[data_index]
                colours = [(1.0, 1.0, 1.0, 0), colour_gen(i)]
                label_cmap = ListedColormap(colours)
                plt.imshow(np.transpose(label_data), cmap=label_cmap, alpha=0.5)
                plt.plot(0, 0, c=colour_gen(i), label=label_name)

        plt.legend(loc=(1.05, 0.8))
        plt.show()