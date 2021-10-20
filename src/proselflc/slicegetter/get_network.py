import torch.nn as nn

from proselflc.exceptions import ParamException
from proselflc.slices.networks.resnet import resnet18, resnet34
from proselflc.slices.networks.shufflenetv2 import shufflenetv2


class NetworkPool:
    """
    Collection for validated networks

    A dictionary of network_name (key) and nn.Module (not initialised).

    TODO:
        Now the networks fix class_num = 100 by default.
        To change this and make it flexible for more use cases.
    """

    validated_networks = {
        "shufflenetv2": shufflenetv2,  # faster than shufflenet
        "resnet18": resnet18,
        "resnet34": resnet34,
    }

    untested_networks = {}

    @classmethod
    def get_network(cls, params={}) -> nn.Module:
        """
        Returns:
            nn.Module, a predefined network archiecture.

        Inputs: A dictionary of params
            params["network_name"]: str = "shufflenetv2", a predefined network name.

        TODO:
            Tested current networks.
            More networks added and tested.
        """

        # sanity check for network_name
        if "network_name" not in params.keys():
            error_msg = (
                "The input params have no key of network_name. "
                + "params["
                + "network_name"
                + "] "
                + " has to be provided."
            )
            raise (ParamException(error_msg))

        if not isinstance(params["network_name"], str):
            error_msg = "The given network_name is not a string."
            raise (ParamException(error_msg))

        if params["network_name"] in cls.validated_networks.keys():
            # TODO: more params to config the returned network
            return cls.validated_networks[params["network_name"]](
                params=params,
            )
        else:
            error_msg = (
                "The given network_name is "
                + params["network_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + str(cls.validated_networks.keys())
            )
            raise (ParamException(error_msg))
