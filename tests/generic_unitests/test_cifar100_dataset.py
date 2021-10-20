import unittest

import numpy as np

from proselflc.slices.datain.datasets.cifar100dataset import CIFAR100Dataset


class TestCIFAR100Dataset(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        pass

    def test_symmetric_noise_rate(self):
        cifar_train_dataset = CIFAR100Dataset(
            params={
                "train": True,
                "symmetric_noise_rate": 0,
            }
        )
        cifar_test_dataset = CIFAR100Dataset(
            params={
                "train": False,
                "symmetric_noise_rate": 0,
            }
        )
        # evaluate configs
        self.assertTrue(cifar_train_dataset.root == "./datasets")
        self.assertTrue(cifar_test_dataset.root == "./datasets")
        self.assertTrue(cifar_train_dataset.train)
        self.assertEqual(cifar_test_dataset.train, False)

        # evaluate content
        print(cifar_train_dataset.__len__())  # len(self.datain)
        print(cifar_test_dataset.__len__())  # len(self.datain)
        # datain
        print(cifar_train_dataset.data.__class__)  # numpy.ndarray
        print(cifar_test_dataset.data.__class__)  # numpy.ndarray
        print(cifar_train_dataset.data.dtype)  # uint 8
        print(cifar_test_dataset.data.dtype)  # uint 8
        print(cifar_train_dataset.data.shape)  # (50000, 32, 32, 3)
        print(cifar_test_dataset.data.shape)  # (10000, 32, 32, 3)
        # target
        print(cifar_train_dataset.targets.__class__)  # list
        print(cifar_test_dataset.targets.__class__)  # list
        print(max(cifar_train_dataset.targets))  # 99
        print(min(cifar_train_dataset.targets))  # 0
        self.assertTrue(
            max(cifar_train_dataset.targets) == max(cifar_test_dataset.targets)
        )
        self.assertTrue(
            min(cifar_train_dataset.targets) == min(cifar_test_dataset.targets)
        )

        cifar_train_noisy = CIFAR100Dataset(
            params={
                "train": True,
                "symmetric_noise_rate": 0.4,
            }
        )
        noise_rate = sum(
            np.array(cifar_train_noisy.targets) != np.array(cifar_train_dataset.targets)
        ) / len(cifar_train_dataset.targets)
        self.assertTrue(noise_rate == 0.4)

        cifar_test_clean = CIFAR100Dataset(
            params={
                "train": False,
                "symmetric_noise_rate": 0.4,
            }
        )
        noise_rate = sum(
            np.array(cifar_test_clean.targets) != np.array(cifar_test_dataset.targets)
        ) / len(cifar_test_dataset.targets)
        self.assertTrue(noise_rate == 0.0)


if __name__ == "__main__":
    unittest.main()
