import torch

# Transforms that implement this class can be represented using an affine matrix
# in homogeneous coords (e.g. flip, rotate, scale, translate).
class AffineMixin:
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init affine mixin')
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

class RandomAffineMixin:
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__(**kwargs)
        print('init random affine mixin')
        print('setting affine true')
        self._is_affine = True

    def get_affine_back_transform(
        self,
        device: torch.device,   # Device is required as matrix multiplications could be performed in this function, e.g. condensing rotation matrices. 
        **kwargs) -> torch.Tensor:
        t = self.freeze()
        return t.get_affine_back_transform(device, **kwargs)

    def get_affine_transform(
        self,
        device: torch.device,   # Device is required as matrix multiplications could be performed in this function, e.g. condensing rotation matrices. 
        **kwargs) -> torch.Tensor:
        t = self.freeze()
        return t.get_affine_transform(device, **kwargs)
