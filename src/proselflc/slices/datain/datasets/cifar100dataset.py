import copy
import os
import pickle
import random
from typing import Callable, Optional

import numpy as np
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
                # The original label itself is not excluded, e.g., DIVIDEMIX.
                #
                # While in the other papers, e.g., ProSelfLC,
                # the orginal label itself is excluded.
                # noisy_target = random.randint(0, class_num-1)
                # This may affect the actual noisy rate when class number is small
                # e.g., ciar10 dataset.
                noisy_label_options = list(range(self.class_num))
                noisy_label_options.pop(self.targets[idx])
                rand_idx = random.randint(0, len(noisy_label_options) - 1)
                self.targets[idx] = noisy_label_options[rand_idx]

        noise_key = "asymmetric_noise_rate_finea2b"
        if params["train"] and noise_key in params.keys():
            # generate noisy labels only for training data
            self.asymmetric_noise_rate = params[noise_key]
            if self.asymmetric_noise_rate < 0 or self.asymmetric_noise_rate > 1:
                error_msg = (
                    "{}:{}, ".format(noise_key, self.asymmetric_noise_rate)
                    + ", has to be in the range [0, 1]"
                )
                raise ParamException(error_msg)
            self.build_coarse2finelabels()

            # Asymmetric label noise: we follow [46] to generate asymmetric label noise
            # to fairly compare with their reported results. Within each
            # coarse class, we randomly select two fine classes A and B.
            # Then we flip r × 100% labels of A to B, and r × 100%
            # labels of B to A. We remark that the overall label noise rate
            # is smaller than r.
            selection_num = 2
            target_arr = np.array(self.targets)
            for coarse_label in self.coarse2finelabels:
                fine_set = self.coarse2finelabels[coarse_label]
                [fine_a, fine_b] = random.sample(fine_set, selection_num)
                noise_dict = {
                    fine_a: fine_b,
                    fine_b: fine_a,
                }
                temp = copy.deepcopy(target_arr)
                for (ori_label, noise_label) in noise_dict.items():
                    index_list = np.where(temp == ori_label)[0].tolist()
                    # shuffle in place
                    random.shuffle(index_list)
                    noise_num = int(self.asymmetric_noise_rate * len(index_list))
                    target_arr[index_list[:noise_num]] = noise_label
            self.targets = target_arr.tolist()

    def build_coarse2finelabels(self) -> None:
        self.coarse2finelabels = {}
        for file_name, checksum in self.test_list:
            # In fact: len(self.test_list) = 1
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                if "fine_labels" in entry and "coarse_labels" in entry:
                    assert len(entry["coarse_labels"]) == len(entry["fine_labels"])
                    for coarse_label, fine_label in zip(
                        entry["coarse_labels"], entry["fine_labels"]
                    ):
                        fine_set = self.coarse2finelabels.get(
                            coarse_label,
                            set(),
                        )
                        fine_set.add(fine_label)
                        self.coarse2finelabels.update(
                            {
                                coarse_label: fine_set,
                            }
                        )
