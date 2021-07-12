import unittest

from proselflc.bricks.datain.datasets.cifar100dataset import CIFAR100Dataset


class TestCIFAR100Dataset(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        pass

    def test_usages_and_content(self):
        cifar_train_dataset = CIFAR100Dataset(train=True)
        cifar_test_dataset = CIFAR100Dataset(train=False)
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


if __name__ == "__main__":
    unittest.main()
