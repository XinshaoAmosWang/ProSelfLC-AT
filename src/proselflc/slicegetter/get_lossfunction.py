import torch.nn as nn

from proselflc.exceptions import ParamException
from proselflc.slices.losses.confidencepenalty import ConfidencePenalty
from proselflc.slices.losses.crossentropy import CrossEntropy
from proselflc.slices.losses.labelcorrection import LabelCorrection
from proselflc.slices.losses.lablesmoothing import LabelSmoothing
from proselflc.slices.losses.mean_absolute_error import MeanAbsoluteError
from proselflc.slices.losses.proselflc import ProSelfLC


class LossPool:
    """
    Collection for validated losses

    A dictionary of loss_name (key) and nn.Module (not initialised).

    """

    validated_losses = {
        "confidencepenalty": ConfidencePenalty,
        "crossentropy": CrossEntropy,
        "labelcorrection": LabelCorrection,
        "lablesmoothing": LabelSmoothing,
        "proselflc": ProSelfLC,
        "dm_exp_pi": MeanAbsoluteError,
    }

    @classmethod
    def get_lossfunction(cls, params={}) -> nn.Module:
        """
        Returns:
            nn.Module, a predefined loss

        Inputs: A dictionary of params
            params["loss_name"]: str = "proselflc", a predefined network name.
            For proselflc:
                params["totoal_time"]: int, which is the total iterations or epochs
                params["exp_base"]: float
                params["counter"]: str, eithor "iteration or "epoch"

        TODO:
            More losses added and tested.
        """

        # sanity check for network_name
        if "loss_name" not in params.keys():
            error_msg = (
                "The input params have no key of loss_name. "
                + "params["
                + "loss_name"
                + "] "
                + " has to be provided."
            )
            raise (ParamException(error_msg))

        if not isinstance(params["loss_name"], str):
            error_msg = "The given loss_name is not a string."
            raise (ParamException(error_msg))

        if params["loss_name"] in cls.validated_losses.keys():
            loss_class = cls.validated_losses[params["loss_name"]]
            return loss_class(params=params)
        else:
            error_msg = (
                "The given loss_name is "
                + params["loss_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + str(cls.validated_losses.keys())
            )
            raise (ParamException(error_msg))
