import torch.nn as nn

class ParotidLeft3DNet(nn.Module):
    def __init__(
        self,
        localiser: nn.Module,
        segmenter: nn.Module):
        """
        args:
            localiser: the localiser 
        """
        super().__init__()

