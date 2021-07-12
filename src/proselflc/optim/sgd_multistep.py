import torch.optim as optim

from proselflc.exceptions import ParamException


class SGDMultiStep:
    """
    Setup SGD optimise with MultiStep learning rate scheduler

    Input:
        1. net.parameters()
        2. params, a dictory of key:value map.
            params["lr"]: float, e.g., 0.1
            params["milestones"]: e.g., [60, 120, 160]
            params["gamma"]: float, e.g., 0.2

    Remark:
        In this version, fix some parmas to make it simpler to use:
            params["momentum"] = 0.9
            params["weight_decay"] = 5e-4

    Return:
        1. optimiser
        2. train_scheduler

    TODO:
        unitests
    """

    def __init__(self, net_params, params):
        """
        Input:
            1. net_params: e.g., net.parameters()
            2. params, a dictory of key:value map.
                params["lr"]: float, e.g., 0.1
                params["milestones"]: e.g., [60, 120, 160]
                params["gamma"]: float, e.g., 0.2
        """
        # TODO: more sanity check
        params["momentum"] = 0.9
        params["weight_decay"] = 5e-4

        if "lr" not in params.keys() or not isinstance(params["lr"], float):
            error_msg = (
                "The input params have no key of lr. "
                + "params["
                + "lr"
                + "] "
                + " has to be provided as a float data type."
            )
            raise (ParamException(error_msg))

        if "milestones" not in params.keys() or not isinstance(
            params["milestones"], list
        ):
            error_msg = (
                "The input params have no key of milestones. "
                + "params["
                + "milestones"
                + "] "
                + " has to be provided as a list of integers."
                + "E.g., params["
                + "milestones"
                + "] = [60, 120, 160]"
            )
            raise (ParamException(error_msg))

        if "gamma" not in params.keys() or not isinstance(params["gamma"], float):
            error_msg = (
                "The input params have no key of gamma. "
                + "params["
                + "gamma"
                + "] "
                + " has to be provided as a float data type."
            )
            raise (ParamException(error_msg))

        self.optimizer = optim.SGD(
            net_params,
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params["weight_decay"],
        )

        # learning rate decay
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=params["milestones"], gamma=params["gamma"]
        )
