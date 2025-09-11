import torch

from .spatial import SpatialTransform

class HomogeneousTransform(SpatialTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_homogeneous = True

    def get_homogeneous_back_transform(
        self,
        *args,
        **kwargs) -> torch.Tensor:
        raise ValueError(f"Subclasses of 'HomogeneousTransform' must implement 'get_homogeneous_back_transform' method.")
