import os
from torch.utils.data import Dataset

ROOT_DIR = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed')

class ImageDataset(Dataset):
    def __init__(self):
        self.patient_summary_df = DatasetSummary().patient_summary()

    def __len__(self):
        """
        returns: number of axial slices in the dataset.
        """
        return self.patient_summary_df['res-z'].sum()

    def find(arr, search_fn):
        """
        returns: the first element that matches the search function, else None.
        arr: the array to search.
        find_fn: the search function.
        """
        for a in arr:
            if search_fn(a):
                return a

        return None

    def __getitem__(self, idx):
        """
        returns: an (input, label) pair from the dataset.
        idx: the item to return.
        """
        # Calculate patient and slice from index.
        pat_id, slice_id = find()