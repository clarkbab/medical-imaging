# Patient objects.

def build_modified():
    # Moves CT/RTSTRUCT files to the standard folder structure.

def list_patients():
    # returns all Patient objects.

# Issues:
# - should be able to handle flat file structures, when no 'pat_id' folders exist.
#   to do this, it needs to read pat_id from dicom metadata.
# - some patients might have CT scans split across multiple folders. we'll need to
#   rely on the pat_id within the folder again.

# datasets contain:
# - 'raw' folder which we should be able to just copy raw data to. should handle nested
#   and flat structures.
# - 'modified' folder which is built the first time a dataset is accessed. this contains
#   CT/RTSTRUCT data that is nested under patient ID folders. the benefit of storing the
#   data in this way is that we don't have to 