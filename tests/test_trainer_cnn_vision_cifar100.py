import os
import time
import unittest

import pandas
import torch

from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.trainer.trainer_cnn_vision import Trainer


class TestTrainer(unittest.TestCase):
    def set_params_vision(self):
        self.params.update(
            {
                "data_name": "cifar100",
                "num_classes": 100,  # 1000
                "network_name": "shufflenet",
            }
        )

    def setUp(self):
        """
        This function is an init for all tests
        """
        self.params = {}
        self.params["device"] = "gpu"
        self.params["num_workers"] = 8
        self.params["batch_size"] = 192

        self.params["loss_name"] = "proselflc"
        self.params["exp_base"] = 10
        # len( self.traindataloader ) = 391 when batch size=128
        # len( self.traindataloader ) = 261 when batch size=192
        self.params["total_time"] = 100 * 261
        self.params["counter"] = "iteration"
        # self.params["epsilon"] = 0.2

        self.params["lr"] = 0.01
        self.params["milestones"] = [50]
        self.params["gamma"] = 0.1

    def test_trainer_cifar100(self):
        self.set_params_vision()

        k = 0
        for self.params["network_name"] in ["shufflenet", "shufflenetv2"]:
            for self.params["exp_base"] in [5, 10, 15]:
                for (self.params["total_time"], self.params["counter"]) in zip(
                    [100 * 261, 100], ["iteration", "epoch"]
                ):
                    for self.params["lr"] in [0.01]:
                        k = k + 1
                        # if k == 1:
                        #    continue
                        dt_string = time.strftime("%Y%m%d-%H%M%S")
                        summary_writer_dir = (
                            self.params["data_name"]
                            + "_"
                            + self.params["network_name"]
                            + "_"
                            + self.params["loss_name"]
                            + "_expbase_"
                            + str(self.params["exp_base"])
                            + "_"
                            + str(self.params["total_time"])
                            + "_"
                            + str(self.params["counter"])
                            + "_"
                            + dt_string
                        )
                        self.params["summary_writer_dir"] = (
                            "/home/xinshao/experiments/"
                            + self.params["data_name"]
                            + "/"
                            + self.params["network_name"]
                            + "/"
                            + summary_writer_dir
                        )
                        if not os.path.exists(self.params["summary_writer_dir"]):
                            os.makedirs(self.params["summary_writer_dir"])

                        trainer = Trainer(params=self.params)
                        self.assertTrue(isinstance(trainer, Trainer))
                        self.assertTrue(isinstance(trainer.optim, SGDMultiStep))

                        self.dataframe = pandas.DataFrame(self.params)
                        self.dataframe.to_csv(
                            self.params["summary_writer_dir"] + "/params.csv",
                            encoding="utf-8",
                            index=False,
                            sep="\t",
                            mode="w",  #
                        )

                        # some more test
                        trainer.train()
                        torch.save(
                            trainer.network,
                            self.params["summary_writer_dir"] + "/model.pt",
                        )


if __name__ == "__main__":
    unittest.main()
