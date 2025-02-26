from mymi.utils import *
from mymi.typing import *

from .mednext import create_mednext_v1
from .unet3d import UNet3D

def layer_summary(
    model: str,
    *args,
    leafs_only: bool = True,
    params_only: bool = False,
    **kwargs) -> pd.DataFrame:
    if model == 'unet3d:m':
        model = UNet3D(*args, **kwargs)
    elif model == 'unet3d:l':
        model = UNet3D(*args, n_features=64, **kwargs)
    elif model == 'unet3d:xl':
        model = UNet3D(*args, n_features=128, **kwargs)
    elif model == 'mednext:s':
        model = create_mednext_v1(*args, 'S', **kwargs)
    elif model == 'mednext:b':
        model = create_mednext_v1(*args, 'B', **kwargs)
    elif model == 'mednext:m':
        model = create_mednext_v1(*args, 'M', **kwargs)
    elif model == 'mednext:l':
        model = create_mednext_v1(*args, 'L', **kwargs)
    else:
        raise ValueError(f'Unknown model: {model}.')

    cols = {
        'module': str,
        'module-type': str,
        'n-params': int,
        'param-shapes': str,
    }
    df = pd.DataFrame(columns=cols.keys())
    for n, m in model.named_modules():
        submodules = list(m.modules())
        if leafs_only and len(submodules) != 1:
            continue

        n_params = len(list(m.parameters()))
        if params_only and n_params == 0:
            continue
            
        param_shapes = []
        for p in m.parameters():
            param_shapes.append(str(list(p.shape)))
        param_shapes = ','.join(param_shapes)

        data = {
            'module': n,
            'module-type': m.__class__.__name__,
            'n-params': n_params,
            'param-shapes': param_shapes,
        }
        df = append_row(df, data)
        
    return df