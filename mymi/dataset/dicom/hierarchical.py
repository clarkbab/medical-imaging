import os
import pydicom as dicom

def hierarchical_exists(self) -> bool:
    """
    returns: True if the hierarchical dataset has been built.
    """
    # Check if folder exists.
    hier_path = os.path.join(self._path, 'hierarchical')
    return os.path.exists(hier_path)

def build_hierarchical(self) -> None:
    """
    effect: creates a hierarchical dataset based on dicom content, not existing structure.
    """
    # Load all dicom files.
    raw_path = os.path.join(self._path, 'raw')
    dicom_files = []
    for root, _, files in os.walk(raw_path):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))

    # Copy dicom files.
    for f in sorted(dicom_files):
        # Get patient ID.
        dcm = dicom.read_file(f)
        pat_id = dcm.PatientID

        # Get modality.
        mod = dcm.Modality.lower()
        if not mod in ('ct', 'rtstruct'):
            continue

        # Create filepath.
        hier_path = os.path.join(self._path, 'hierarchical')
        filename = os.path.basename(f)
        filepath = os.path.join(hier_path, pat_id, mod, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save dicom.
        dcm.save_as(filepath)

def require_hierarchical(fn):
    """
    effect: ensures that the hierarchical data
    args:
        fn: the wrapped function.
    """
    def wrapper_fn(self, *args, **kwargs):
        if not hierarchical_exists(self):
            build_hierarchical(self)
        return fn(self, *args, **kwargs)
    return wrapper_fn
