import unittest

import torch
from torch.utils.data import DataLoader

from proselflc.slices.datain.dataloaders.cifar100dataloader import CIFAR100DataLoader


class TestCIFAR100DataLoader(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        pass

    def test_constructor(self):
        cifar_trainloader = CIFAR100DataLoader(
            train=True,
            batch_size=1,
            num_workers=1,
        )
        cifar_testloader = CIFAR100DataLoader(
            train=False,
            batch_size=1,
            num_workers=1,
        )
        # evaluate
        self.assertTrue(isinstance(cifar_trainloader, DataLoader))
        self.assertTrue(isinstance(cifar_testloader, DataLoader))
        #
        self.assertEqual(cifar_trainloader.num_workers, 1)
        self.assertEqual(cifar_trainloader.batch_size, 1)

        for dataloader in [cifar_trainloader, cifar_testloader]:
            for batch_index, (inputs, labels) in enumerate(dataloader):
                print(inputs)
                print(torch.max(inputs))
                print(torch.min(inputs))
                print(inputs.shape)
                print(labels)
                return


if __name__ == "__main__":
    unittest.main()
