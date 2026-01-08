from typing import *

from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..identity import Identity
from .intensity import IntensityTransform
from .random import RandomIntensityTransform

# Min:
# m=0.2 -> m=(0.2, 0.2, ...)
# m=(-0.2, 0.2) -> m=(-0.2, 0.2, ...)
# Max:
# m=1.5 -> m=
class RandomRescale(RandomIntensityTransform):
    def __init__(
        self,
        min_range: Union[Number, Tuple[Number, ...]] = 0,
        max_range: Union[Number, Tuple[Number, ...]] = 1,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.__min_range = to_tensor(min_range, broadcast=2)
        assert len(self.__min_range) == 2, f"Expected 'min_range' of length 2, got {len(self.__min_range)}."
        self.__max_range = to_tensor(max_range, broadcast=2)
        assert len(self.__max_range) == 2, f"Expected 'max_range' of length 2, got {len(self.__max_range)}."
        self._params = dict(
            dim=self._dim,
            max_range=self.__max_range,
            min_range=self.__min_range,
            p=self._p,
        )

    def freeze(self) -> 'Norm':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random(2))
        min_draw = draw[0] * (self.__min_range[1] - self.__min_range[0]) + self.__min_range[0]
        max_draw = draw[0] * (self.__max_range[1] - self.__max_range[0]) + self.__max_range[0]
        params = dict(
            min=min_draw,
            max=max_draw,
        )
        return super().freeze(Rescale, params)

    def __str__(self) -> str:
        params = dict(
            min_range=to_tuple(self.__min_range),
            max_range=to_tuple(self.__max_range),
        )
        return super().__str__(self.__class__.__name__, params)

class Rescale(IntensityTransform):
    def __init__(
        self,
        min: Number = 0,
        max: Number = 1,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.__min = min
        self.__max = max
        self._params = dict(
            dim=self._dim,
            max=self.__max,
            min=self.__min,
        )

    def __str__(self) -> str:
        params = dict(
            min=self.__min,
            max=self.__max,
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_intensity(
        self,
        image: ImageTensor,
        ) -> ImageTensor:
        if image.dtype == torch.bool:
            return image    # Boolean tensors are unchanged by intensity transforms. 
        print('rescaling')
        print(image.shape)
        print(self.__min, self.__max)
        print(image.min(), image.max())
        image_t = (self.__max - self.__min) * (image - image.min()) / (image.max() - image.min()) + self.__min
        print(image_t.min(), image_t.max())
        print(image_t.shape)
        return image_t
