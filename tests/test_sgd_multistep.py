import unittest

import torch.optim as optim

from proselflc.exceptions import ParamException
from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.slicegetter.get_network import NetworkPool

get_network = NetworkPool.get_network


class TestSGDMultiStep(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        pass

    def test_sgd_multistep(self):
        params = {}
        params["lr"] = 0.1
        params["milestones"] = [60, 120, 160]
        params["gamma"] = 0.2
        params["network_name"] = "shufflenetv2"

        net = get_network(params=params)
        sgd_multistep_optimiser = SGDMultiStep(net.parameters(), params=params)

        self.assertTrue(isinstance(sgd_multistep_optimiser.optimizer, optim.SGD))
        self.assertTrue(
            isinstance(
                sgd_multistep_optimiser.lr_scheduler, optim.lr_scheduler.MultiStepLR
            )
        )

        with self.assertRaises(ParamException):
            params["gamma"] = ""
            SGDMultiStep(net.parameters(), params=params)
        with self.assertRaises(ParamException):
            params["milestones"] = ""
            SGDMultiStep(net.parameters(), params=params)
        with self.assertRaises(ParamException):
            params["lr"] = ""
            SGDMultiStep(net.parameters(), params=params)


if __name__ == "__main__":
    unittest.main()
