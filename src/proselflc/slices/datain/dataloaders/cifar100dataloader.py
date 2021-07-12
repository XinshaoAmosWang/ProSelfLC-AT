from torch.utils.data import DataLoader

from ..datasets.cifar100dataset import CIFAR100Dataset
from ..transforms.cifar100transforms import (
    CIFAR100_transform_test_data,
    CIFAR100_transform_train_data,
)


class CIFAR100DataLoader(DataLoader):
    """
    CIFAR100 Dataloader with customed settings.

    What is special here versus DataLoader:
        1. train is bool and required.
            1.1. which dataset
            1.2. shuffle=train accordingly.
            1.3. set data_transform and target_tranform accordingly.

    Args:
        train (bool, required):
            If true, it is a training dataloader.
            Otherwise, it is a testing dataloader
            1. shuffle(bool, not required):
                It is hidden in this class.
                being equeal to train (bool, required).
            2. which dataset will be set accordingly.
            3. transform will be set accordingly.
        num_workers:
            inherited from DataLoader.
        batch_size:
            inherited from DataLoader
    """

    # overwrite
    def __init__(
        self,
        train: bool,
        num_workers: int = 4,
        batch_size: int = 128,  # to expose this to params
    ) -> None:
        if train:
            self._dataset = CIFAR100Dataset(
                train=train,  # train set or test set
                data_transform=CIFAR100_transform_train_data,
                target_transform=None,
            )
        else:
            self._dataset = CIFAR100Dataset(
                train=train,  # train set or test set
                data_transform=CIFAR100_transform_test_data,
                target_transform=None,
            )

        super().__init__(
            dataset=self._dataset,
            shuffle=train,  # only if train, shuffle.
            num_workers=num_workers,
            batch_size=batch_size,
        )
