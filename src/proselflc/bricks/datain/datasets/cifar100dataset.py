from typing import Callable, Optional

import torchvision


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    """
    CIFAR100 class inherits torchvision.datasets.CIFAR100.

    What is special in this inherited subclass:
        1. root="./datasets"
        2. download=False
        3. rename transform to data_transform

    Args:
        train (bool, required):
            If True, creates dataset from training set,
            otherwise creates from test set.
        data_transform (callable, optional):
            A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the
            target and transforms it.

    Usages:
        cifar_train_dataset=
            CIFAR100(train=True, data_transform=None or something,
                target_tranforms=None or something)
        cifar_test_dataset=
            CIFAR100(train=False, data_transform=None or something,
                target_tranforms=None or something)
    Dataset formats:
        train's datain: (50000, 32, 32, 3)
        test's datain: (10000, 32, 32, 3)
        target: list of values in the range of [0, 99]
    """

    # overwrite
    def __init__(
        self,
        train: bool,
        data_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = "./datasets"
        super().__init__(
            root=self.root,  # fix it
            download=True,  # fix it
            train=train,
            transform=data_transform,
            target_transform=target_transform,
        )
