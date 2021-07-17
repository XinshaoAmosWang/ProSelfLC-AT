from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.slices.datain.dataloaders.cifar100dataloader import CIFAR100DataLoader


class DataLoaderPool:
    """
    Collection for validated data loaders

    A dictionary of data_name (key) and DataLoader (not initialised).

    """

    validated_dataloaders = {
        "cifar100": CIFAR100DataLoader,
    }

    @classmethod
    def get_dataloader(cls, params={}) -> DataLoader:
        """
        Returns:
            DataLoader, preprocessed, iterable and directly feeded into network

        Inputs: A dictionary of params
            params["data_name"]: str = "cifar100", a predefined dataset name.
            params["train"]: bool, true or false
            params["num_workers"]: int
            params["batch_size"]: int

        TODO:
            More dataloaders added and tested.
        """

        # sanity check for params["data_name"]
        if "data_name" not in params.keys():
            error_msg = (
                "The input params have no key of data_name. "
                + "params["
                + "data_name"
                + "] "
                + " has to be provided."
            )
            raise (ParamException(error_msg))

        if not isinstance(params["data_name"], str):
            error_msg = "The given data_name is not a string."
            raise (ParamException(error_msg))
        # sanity check for params["train"]
        if "train" not in params.keys():
            error_msg = (
                "The input params have no key of train. "
                + "params["
                + "train"
                + "] "
                + " has to be provided."
            )
            raise (ParamException(error_msg))

        if not isinstance(params["train"], bool):
            error_msg = "The given train is not a bool type."
            raise (ParamException(error_msg))

        if params["data_name"] in cls.validated_dataloaders.keys():
            dataloader_class = cls.validated_dataloaders[params["data_name"]]
            # num_workers, batch size are not well sanity checked.
            return dataloader_class(params)
        else:
            error_msg = (
                "The given data_name is "
                + params["data_name"]
                + ", which is not supported yet."
                + "Please choose from "
                + str(cls.validated_dataloaders.keys())
            )
            raise (ParamException(error_msg))
