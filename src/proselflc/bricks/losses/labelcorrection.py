from torch import Tensor

from .crossentropy import CrossEntropy


class LabelCorrection(CrossEntropy):
    """
    The implementation for label correction.
    The target probability will be smoothed by
    a predicted distributions, i.e., self knowledge.
        1. a.k.a. Bootsoft
        2. a.k.a. Joint-soft
        3. share principle with Pesudo-labelling

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. epsilon, which controls the degree of label correction.

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
            epsilon: which controls the degree of label correction.

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if epsilon is None:
            epsilon = self.epsilon
        new_target_probs = (1 - epsilon) * target_probs + epsilon * pred_probs
        # reuse CrossEntropy's forward computation
        return super().forward(pred_probs, new_target_probs)
