import torch

from mymi.utils import *
from mymi.typing import *

from .mednext import create_mednext_v1
from .unet import UNet2D, UNet3D

def get_model(
    arch: str,
    n_output_channels: int,
    **kwargs,
    ) -> torch.nn.Module:
    if arch == 'unet2d:m':
        logging.info(f"Using UNet2D (M) with {n_output_channels} channels.")
        return UNet2D(n_output_channels, **kwargs)
    elif arch == 'unet2d:l':
        logging.info(f"Using UNet2D (L) with {n_output_channels} channels.")
        return UNet2D(n_output_channels, n_features=64, **kwargs)
    elif arch == 'unet2d:xl':
        logging.info(f"Using UNet2D (XL) with {n_output_channels} channels.")
        return UNet2D(n_output_channels, n_features=128, **kwargs)
    elif arch == 'unet3d:m':
        logging.info(f"Using UNet3D (M) with {n_output_channels} channels.")
        return UNet3D(n_output_channels, **kwargs)
    elif arch == 'unet3d:l':
        logging.info(f"Using UNet3D (L) with {n_output_channels} channels.")
        return UNet3D(n_output_channels, n_features=64, **kwargs)
    elif arch == 'unet3d:xl':
        logging.info(f"Using UNet3D (XL) with {n_output_channels} channels.")
        return UNet3D(n_output_channels, n_features=128, **kwargs)
    elif arch == 'mednext:s':
        logging.info(f"Using MedNeXt (S) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'S', **kwargs)
    elif arch == 'mednext:b':
        logging.info(f"Using MedNeXt (B) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'B', **kwargs)
    elif arch == 'mednext:m':
        logging.info(f"Using MedNeXt (M) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'M', **kwargs)
    elif arch == 'mednext:l':
        logging.info(f"Using MedNeXt (L) with {n_output_channels} channels.")
        return create_mednext_v1(1, n_output_channels, 'L', **kwargs)
    else:
        raise ValueError(f"Unknown architecture '{arch}'.")

def layer_summary(
    arch: str,
    n_output_channels: int,
    leafs_only: bool = True,
    params_only: bool = False,
    **kwargs,
    ) -> pd.DataFrame:
    model = get_model(arch, n_output_channels, **kwargs)

    # Summarise layers.
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
