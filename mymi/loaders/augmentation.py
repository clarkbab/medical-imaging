import numpy as np
import SimpleITK as sitk
import torchio as tio
from typing import *

from typing import Optional, Tuple

from mymi import logging
from mymi.transforms import Standardise

class RandomAffine():
    def __init__(
        self,
        rotation: Tuple[float, float] = (-5, 5),
        scaling: Tuple[float, float] = (0.8, 1.2),
        thresh_high: Optional[float] = None,
        thresh_low: Optional[float] = None,
        translation: Tuple[float, float] = (-50, 50),
        use_elastic: bool = False,
        use_stand: bool = False,
        use_thresh: bool = False,
        **kwargs) -> None:
        # Handle arguments.
        if len(rotation) == 2:
            rotation = rotation * 3
        assert len(rotation) == 6
        self.__rotation = rotation
        if len(scaling) == 2:
            scaling = scaling * 3
        assert len(scaling) == 6
        self.__scaling = scaling
        if len(translation) == 2:
            translation = translation * 3
        assert len(translation) == 6
        self.__translation = translation

        logging.info(f"Using sitk augmentation: rotation={self.__rotation}, scaling={self.__scaling}, translation={self.__translation}.")

    def get_concrete_transform(
        self,
        random_seed: Optional[int] = None) -> Tuple[sitk.Transform, sitk.Transform, Dict[str, Any]]:
        # Sample transform parameters.
        if random_seed is not None:
            np.random.seed(random_seed)
        scaling = [np.random.uniform(*self.__scaling[2 * i:2 * i + 2]) for i in range(3)]
        rotation = [np.random.uniform(*self.__rotation[2 * i:2 * i + 2]) for i in range(3)]
        translation = [np.random.uniform(*self.__translation[2 * i:2 * i + 2]) for i in range(3)]
        # logging.info(f"Concrete params: rotation={rotation}, scaling={scaling}, translation={translation}.")

        # Create transform.
        scaling_transform = sitk.ScaleTransform(3)
        scaling_transform.SetScale(scaling)
        rigid_transform = sitk.Euler3DTransform()
        rotation_rad = np.radians(rotation).tolist()
        rigid_transform.SetRotation(*rotation_rad)
        rigid_transform.SetTranslation(translation)
        transforms = [scaling_transform, rigid_transform]
        # Our scaling/rot/trans arguments should represent the forward transform (moving -> fixed) space,
        # as this would be what the users expect. We invert this to get the backward transform used
        # for resampling.
        forward_transform = sitk.CompositeTransform(transforms)
        backward_transform = forward_transform.GetInverse()

        # Return params for debugging.
        params = {
            'rotation': rotation,
            'scaling': scaling,
            'translation': translation
        }

        return forward_transform, backward_transform, params

def get_transforms(
    rotation: Tuple[float, float] = (-5, 5),
    scale: Tuple[float, float] = (0.8, 1.2),
    thresh_high: Optional[float] = None,
    thresh_low: Optional[float] = None,
    translation: Tuple[float, float] = (-50, 50),
    use_elastic: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    **kwargs) -> Tuple[tio.Transform, tio.Transform]:
    logging.info(f"Using augmentation: rotation={rotation}, scale={scale}, translation={translation}.")

    # Create transforms.
    transform_train = tio.transforms.RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')
    transform_val = None

    if use_elastic:
        transform_train = tio.transforms.Compose([
           transform_train,
           tio.transforms.RandomElasticDeformation() 
        ])

    if use_thresh:
        transform_train = tio.transforms.Compose([
            transform_train,
            tio.transforms.Clamp(out_min=thresh_low, out_max=thresh_high)
        ])
        transform_val = tio.transforms.Clamp(out_min=thresh_low, out_max=thresh_high)

    if use_stand:
        stand = Standardise(-832.2, 362.1)
        transform_train = tio.transforms.Compose([
            transform_train,
            stand
        ])
        if transform_val is None:
            transform_val = stand
        else:
            transform_val = tio.transforms.Compose([
                transform_val,
                stand
            ])

    return transform_train, transform_val
