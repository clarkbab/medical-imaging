
class ProcessedDataset:
    ###
    # Subclasses must implement.
    ###

    @classmethod
    def data_dir(cls):
        raise NotImplementedError("Method 'data_dir' not implemented in subclass.")

    ###
    # Basic queries.
    ###