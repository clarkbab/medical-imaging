from torch import nn
from typing import Optional

class LayerWrapper(nn.Module):
    def __init__(self, layer):
        super(LayerWrapper, self).__init__()
        self.__layer = layer

    @property
    def out_channels(self) -> Optional[int]:
        if hasattr(self.__layer, 'out_channels'):
            return self.__layer.out_channels
        else:
            return None
