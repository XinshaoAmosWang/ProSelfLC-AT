import unittest

from proselflc.exceptions import ParamException
from proselflc.slicegetter.get_lossfunction import LossPool
from proselflc.slices.losses.confidencepenalty import ConfidencePenalty
from proselflc.slices.losses.crossentropy import CrossEntropy
from proselflc.slices.losses.labelcorrection import LabelCorrection
from proselflc.slices.losses.lablesmoothing import LabelSmoothing
from proselflc.slices.losses.proselflc import ProSelfLC

get_lossfunction = LossPool.get_lossfunction


class TestGetLossfunction(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        pass

    def test_getlossfunction(self):
        params = {}
        params["loss_name"] = "some_loss"
        with self.assertRaises(ParamException):
            get_lossfunction(params)

        with self.assertRaises(ParamException):
            get_lossfunction({})
        with self.assertRaises(ParamException):
            get_lossfunction({"no loss_name": "crossentropy"})
        with self.assertRaises(ParamException):
            get_lossfunction({"loss_name": 123})

        params["epsilon"] = 0.0
        params["loss_name"] = "confidencepenalty"
        self.assertTrue(isinstance(get_lossfunction(params), ConfidencePenalty))
        params["loss_name"] = "crossentropy"
        self.assertTrue(isinstance(get_lossfunction(params), CrossEntropy))
        params["loss_name"] = "labelcorrection"
        self.assertTrue(isinstance(get_lossfunction(params), LabelCorrection))
        params["loss_name"] = "lablesmoothing"
        self.assertTrue(isinstance(get_lossfunction(params), LabelSmoothing))

        params["loss_name"] = "proselflc"
        params["total_epochs"] = 200
        params["counter"] = "epoch"
        params["exp_base"] = 1
        self.assertTrue(isinstance(get_lossfunction(params), ProSelfLC))


if __name__ == "__main__":
    unittest.main()
