import torch

from mymi import logging

class SpatialSmoothingLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logging.info(f"Initialising SpatialSmoothingLoss.")

    def forward(
        self,
        pred: torch.Tensor) -> torch.Tensor:
        # Pred shape: N, 3, X, Y, Z.
        shape = pred.shape
        # assert len(shape) == 5
        ndims = shape[1]
        assert len(shape) == ndims + 2
        # assert shape[1] == 3
        
        # Create x/y/z partials.
        d_preds = []
        for i in range(shape[1]):
            # Get spatial diff.
            pred_p1 = pred.roll(-1, dims=i + 2)
            d_pred = pred_p1 - pred

            # Set edge voxels to zero.
            index = [slice(None)] * len(shape)
            index[i + 2] = -1
            d_pred[index] = 0
            d_preds.append(d_pred)

        # D_pred shape: N, 9, X, Y, Z.
        d_pred = torch.stack(d_preds).movedim(0, 1)
        d_pred = d_pred.reshape((shape[0], ndims * ndims, *shape[2:]))
        
        # Calculate norm loss.
        norms = d_pred ** 2
        loss = norms.mean()
        
        return loss
