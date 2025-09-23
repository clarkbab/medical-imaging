import torch

from .spatial import SpatialTransform

class RandomAffine():
    pass

# Flip, Rotation, Translation (and others) should probably subclass this.
class Affine(SpatialTransform):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init affine transform')
        print('setting affine true')
        self._is_affine = True

    def get_affine_back_transform(
        self,
        device: torch.device,   # Device is required as matrix multiplications could be performed in this function, e.g. condensing rotation matrices. 
        **kwargs) -> torch.Tensor:
        raise ValueError(f"Classes with 'AffineMixin' must implement 'get_affine_back_transform' method.")

    def get_affine_transform(
        self,
        device: torch.device,   # Device is required as matrix multiplications could be performed in this function, e.g. condensing rotation matrices. 
        **kwargs) -> torch.Tensor:
        raise ValueError(f"Classes with 'AffineMixin' must implement 'get_affine_transform' method.")
