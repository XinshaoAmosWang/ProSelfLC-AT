import random
from typing import Callable, Optional

import torchvision

from proselflc.exceptions import ParamException


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
        params,
        data_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = "./datasets"

        if "train" not in params.keys():
            error_msg = (
                "{} is a required input param".format("train")
                + ", but is not provided."
            )
            raise (ParamException(error_msg))

        super().__init__(
            root=self.root,  # fix it
            download=True,  # fix it
            train=params["train"],
            transform=data_transform,
            target_transform=target_transform,
        )

        if params["train"] and "symmetric_noise_rate" in params.keys():
            # generate symmetric noisy labels only for training data
            self.symmetric_noise_rate = params["symmetric_noise_rate"]
            if self.symmetric_noise_rate < 0 or self.symmetric_noise_rate > 1:
                error_msg = (
                    "symmetric_noise_rate:{}, ".format(self.symmetric_noise_rate)
                    + ", has to be in the range [0, 1]"
                )
                raise ParamException(error_msg)

            sample_num = self.__len__()
            self.class_num = len(set(self.targets))
            sample_idx_list = list(range(sample_num))
            # shuffle in place
            random.shuffle(sample_idx_list)
            self.noise_num = int(self.symmetric_noise_rate * sample_num)
            self.noise_sample_idx_list = sample_idx_list[: self.noise_num]

            for idx in self.noise_sample_idx_list:
                # NOTE:
                # the original label itself is not excluded, e.g., DIVIDEMIX
                # in other papers, e.g., ProSelfLC,
                # the orginal label itself is excluded.
                # noisy_target = random.randint(0, class_num-1)
                # This may affect the actual noisy rate when class number is small
                # e.g., ciar10 dataset.
                noisy_label_options = list(range(self.class_num))
                noisy_label_options.pop(self.targets[idx])
                rand_idx = random.randint(0, len(noisy_label_options) - 1)
                self.targets[idx] = noisy_label_options[rand_idx]
