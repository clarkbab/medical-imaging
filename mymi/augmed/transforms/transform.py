from typing import *

class Transform:
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def __repr__(self) -> str:
        return str(self)

class DetTransform(Transform):
    pass

class RandomTransform(Transform):
    pass
