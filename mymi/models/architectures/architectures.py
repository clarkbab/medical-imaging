import torch

from mymi.utils import *
from mymi.typing import *

from .mednext import create_mednext_v1
from .unet3d import UNet3D

def get_module(
    arch: str,
    n_output_channels: int) -> torch.nn.Module:
    if arch == 'unet3d:m':
        logging.info(f"Using UNet3D (M) with {n_output_channels} channels.")
        return UNet3D(n_output_channels)
    elif arch == 'unet3d:l':
        logging.info(f"Using UNet3D (L) with {n_output_channels} channels.")
        return UNet3D(n_output_channels, n_features=64)
    elif arch == 'unet3d:xl':
        logging.info(f"Using UNet3D (XL) with {n_output_channels} channels.")
        return UNet3D(n_output_channels, n_features=128)
    elif arch == 'mednext:s':
        logging.info(f"Using MedNeXt (S) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'S')
    elif arch == 'mednext:b':
        logging.info(f"Using MedNeXt (B) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'B')
    elif arch == 'mednext:m':
        logging.info(f"Using MedNeXt (M) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'M')
    elif arch == 'mednext:l':
        logging.info(f"Using MedNeXt (L) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'L')
    else:
        raise ValueError(f"Unknown architecture '{arch}'.")

def layer_summary(
    arch: str,
    *args,
    leafs_only: bool = True,
    params_only: bool = False,
    **kwargs) -> pd.DataFrame:
    if arch == 'unet3d:m':
        model = UNet3D(*args, **kwargs)
    elif arch == 'unet3d:l':
        model = UNet3D(*args, n_features=64, **kwargs)
    elif arch == 'unet3d:xl':
        model = UNet3D(*args, n_features=128, **kwargs)
    elif arch == 'mednext:s':
        model = create_mednext_v1(*args, 'S', **kwargs)
    elif arch == 'mednext:b':
        model = create_mednext_v1(*args, 'B', **kwargs)
    elif arch == 'mednext:m':
        model = create_mednext_v1(*args, 'M', **kwargs)
    elif arch == 'mednext:l':
        model = create_mednext_v1(*args, 'L', **kwargs)
    else:
        raise ValueError(f'Unknown arch: {arch}.')

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
