from torchio import Transform
from torchio.transforms import Clamp, Compose, RandomAffine, RandomElasticDeformation, ZNormalization
from typing import Optional, Tuple

from mymi import logging
from mymi.transforms import Standardise

def get_transforms(
    rotation: Tuple[float, float] = (-5, 5),
    scale: Tuple[float, float] = (0.8, 1.2),
    thresh_high: Optional[float] = None,
    thresh_low: Optional[float] = None,
    translation: Tuple[float, float] = (-50, 50),
    use_elastic: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    **kwargs) -> Tuple[Transform, Transform]:
    logging.info(f"Using augmentation: rotation={rotation}, scale={scale}, translation={translation}.")

    # Create transforms.
    transform_train = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')
    transform_val = None

    if use_elastic:
        transform_train = Compose([
           transform_train,
           RandomElasticDeformation() 
        ])

    if use_thresh:
        transform_train = Compose([
            transform_train,
            Clamp(out_min=thresh_low, out_max=thresh_high)
        ])
        transform_val = Clamp(out_min=thresh_low, out_max=thresh_high)

    if use_stand:
        stand = Standardise(-832.2, 362.1)
        transform_train = Compose([
            transform_train,
            stand
        ])
        if transform_val is None:
            transform_val = stand
        else:
            transform_val = Compose([
                transform_val,
                stand
            ])

    return transform_train, transform_val
