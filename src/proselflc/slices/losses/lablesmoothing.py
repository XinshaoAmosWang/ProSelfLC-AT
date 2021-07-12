from torch import Tensor

from .crossentropy import CrossEntropy


class LabelSmoothing(CrossEntropy):
    """
    The implementation for label smoothing.
    The target probability will be smoothed by
    a uniform distribution, defined by the total class number.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. epsilon, which controls the degree of smoothing

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(self, params: dict = None) -> None:
        super().__init__()
        self.epsilon = params["epsilon"]

    def forward(
        self, pred_probs: Tensor, target_probs: Tensor, epsilon: float = None
    ) -> Tensor:
        """
        Inputs:
            pred_probs: predictions of shape (N, C).
            target_probs: targets of shape (N, C).
            epsilon: which controls the degree of smoothing

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if epsilon is None:
            epsilon = self.epsilon
        class_num = pred_probs.shape[1]
        new_target_probs = (1 - epsilon) * target_probs + epsilon * 1.0 / class_num
        # reuse CrossEntropy's forward computation
        return super().forward(pred_probs, new_target_probs)
