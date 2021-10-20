import torch
from torch import Tensor

from proselflc.exceptions import ParamException

from .crossentropy import CrossEntropy


class ProSelfLC(CrossEntropy):
    """
    The implementation for progressive self label correction (CVPR 2021 paper).
    The target probability will be corrected by
    a predicted distributions, i.e., self knowledge.
        1. ProSelfLC is partially inspired by prior related work,
            e.g., Pesudo-labelling.
        2. ProSelfLC is partially theorectically bounded by
            early stopping regularisation.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. current time (epoch/iteration counter).
        4. total time (total epochs/iterations)
        5. exp_base: the exponential base for adjusting epsilon
        6. counter: iteration or epoch counter versus total time.

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(
        self,
        params: dict = None,
    ) -> None:
        super().__init__()
        self.total_epochs = params["total_epochs"]
        self.exp_base = params["exp_base"]
        self.counter = params["counter"]
        self.epsilon = None
        self.transit_time_ratio = params["transit_time_ratio"]

        if not (self.exp_base >= 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero. "
            )
            raise (ParamException(error_msg))

        if not (isinstance(self.total_epochs, int) and self.total_epochs > 0):
            error_msg = (
                "self.total_epochs = "
                + str(self.total_epochs)
                + ". "
                + "The total_epochs has to be a positive integer. "
            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))

        if "total_iterations" in params.keys():
            # only exist when counter == "iteration"
            self.total_iterations = params["total_iterations"]

    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[1]
            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 1
            )
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform
            # the trade-off
            self.epsilon = global_trust * example_trust
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]

    def forward(
        self, pred_probs: Tensor, target_probs: Tensor, cur_time: int
    ) -> Tensor:
        """
        Inputs:
            1. predicted probability distributions of shape (N, C)
            2. target probability  distributions of shape (N, C)
            3. current time (epoch/iteration counter).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        # update self.epsilon
        self.update_epsilon_progressive_adaptive(pred_probs, cur_time)

        new_target_probs = (1 - self.epsilon) * target_probs + self.epsilon * pred_probs
        # reuse CrossEntropy's forward computation
        return super().forward(pred_probs, new_target_probs)
