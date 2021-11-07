import os
import time
import unittest
from itertools import product

import pandas
import torch

from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.trainer.trainer_cnn_vision import Trainer


class TestTrainer(unittest.TestCase):
    WORK_DIR = None

    def set_params_vision(self):
        self.params.update(
            {
                "data_name": "cifar100",
                "num_classes": 100,
                "device": "gpu",
                #
                "num_workers": 4,
                "batch_size": 128,
            }
        )

    def setUp(self):
        """
        This function is an init for all tests
        """
        self.params = {}
        self.set_params_vision()
        # only for palceholder,
        # since we implement drivative manipulation and IMAE instead.
        self.params["counter"] = "iteration"

        self.params["lr"] = 0.1
        self.params["total_epochs"] = 200
        self.params["eval_interval"] = 4
        self.params["milestones"] = [60, 120, 160]
        self.params["gamma"] = 0.2

    def test_trainer_cifar100(self):
        k = 0
        # we implement drivative manipulation and IMAE instead.
        self.params["loss_name"] = "dm_exp_pi"

        for (
            self.params["symmetric_noise_rate"],
            self.params["network_name"],
            self.params["batch_size"],
            self.params["dm_lambda"],
            self.params["warmup_epochs"],
            self.params["dm_beta"],
        ) in product(
            [
                0.0,
                0.2,
                0.4,
                0.6,
            ],
            ["shufflenetv2", "resnet18", "resnet34"],
            [128],
            [0.0, 0.25, 0.5, 1],
            [16, 4],
            [6, 9, 12, 15],
        ):
            k = k + 1
            print(k)

            dt_string = time.strftime("%Y%m%d-%H%M%S")
            summary_writer_dir = (
                "{:0>3}_".format(k)
                + self.params["loss_name"]
                + "_warm"
                + str(self.params["warmup_epochs"])
                + "_lambda"
                + str(self.params["dm_lambda"])
                + "_beta"
                + str(self.params["dm_beta"])
                + "_"
                + dt_string
            )
            self.params["summary_writer_dir"] = (
                self.WORK_DIR
                + "/"
                + self.params["data_name"]
                + "_symmetric_noise_rate_"
                + str(self.params["symmetric_noise_rate"])
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

            trainer.train()
            torch.save(
                trainer.network,
                self.params["summary_writer_dir"] + "/model.pt",
            )


if __name__ == "__main__":

    work_dir = os.getenv(
        "SM_CHANNEL_WORK_DIR",
        "/home/xinshao/proselflc_experiments/",
    )
    TestTrainer.WORK_DIR = work_dir

    print(TestTrainer.WORK_DIR)

    unittest.main()
