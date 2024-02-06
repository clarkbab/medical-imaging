import torch
from torch.nn.functional import grid_sample

# Remember that the DVF actually displaces the output space
# (i.e. the resampling grid), not the input space.
def apply_dvf(
    image: torch.Tensor,
    dvf: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image (torch.Tensor): Image of shape (N, C, X, Y, Z).
        dvf (torch.Tensor): DVF of shape (N, 3, X, Y, Z).
    """
    assert image.shape[0] == 1

    # Create "identity" DVF.
    size = image.shape[2:]
    grid = torch.meshgrid(*(torch.linspace(-1, 1, s) for s in size))
    grid = torch.stack(tuple(reversed(grid)), dim=-1).type(torch.float)
    grid = grid.unsqueeze(0).to(image.device)
    
    # Add deformation.
    dvf = dvf.movedim(1, -1)
    if not dvf.shape == grid.shape:
        raise ValueError(f"DVF shape '{dvf.shape}' does not match required shape '{grid.shape}'.")
    grid = grid + dvf

    # Perform sampling.
    image = image.type(torch.float64)
    grid = grid.type(torch.float64)
    output = grid_sample(image, grid)

    return output
