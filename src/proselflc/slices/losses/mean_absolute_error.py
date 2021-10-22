import torch
import torch.nn as nn
from torch import Tensor

from proselflc.exceptions import ParamException


class MeanAbsoluteError(nn.Module):
    """
    The new implementation of MAE using two distributions.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(self, params: dict = None) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, pred_probs: Tensor, target_probs: Tensor) -> Tensor:
        """
        Inputs:
            pred_probs: predictions of shape (N, C).
            target_probs: targets of shape (N, C).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if not (pred_probs.shape == target_probs.shape):
            error_msg = (
                "pred_probs.shape = " + str(pred_probs.shape) + ". "
                "target_probs.shape = "
                + str(target_probs.shape)
                + ". "
                + "Their shape has to be identical. "
            )
            raise (ParamException(error_msg))
        # TODO: to assert values in the range of [0, 1]

        num_examples = pred_probs.shape[0]
        loss = torch.sum(torch.abs(pred_probs - target_probs), 1)
        loss = torch.sum(loss) / num_examples
        return loss
