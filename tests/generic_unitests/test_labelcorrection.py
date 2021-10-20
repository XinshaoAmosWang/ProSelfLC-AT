import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from proselflc.slicegetter.get_lossfunction import LossPool
from proselflc.trainer.utils import logits2probs_softmax

get_lossfunction = LossPool.get_lossfunction


def my_model(x, input_dim, w_init, b):
    bs = x.shape[0]
    return x.reshape(bs, input_dim) @ w_init + b


# #############################
# function to extract grad
def set_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


class TestLossFunction(unittest.TestCase):
    def setUp(self):
        """
        This function is an init for all tests

        For examples:
            1. A unified constructor for an object.
            2. Some common preprocess.
        """
        self.params = {}
        self.params["loss_name"] = "crossentropy"
        self.loss_criterion1 = get_lossfunction(self.params)
        self.params["loss_name"] = "labelcorrection"
        self.params["epsilon"] = 0.0
        self.loss_criterion2 = get_lossfunction(self.params)

        self.params["epsilon"] = 0.25
        self.loss_criterion3 = get_lossfunction(self.params)

    def test_epsilon0_0(self):
        # FashionMNIST Datasets for training/test
        trn_ds = datasets.FashionMNIST(
            "./datasets",
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        # Dataloader for training/test
        batch_size = 4
        trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)

        # paramaters initialization
        input_dim = 784  # 28x28 FashionMNIST data
        output_dim = 10
        w_init = np.random.normal(scale=0.05, size=(input_dim, output_dim))
        w_init = torch.tensor(w_init, requires_grad=True).float()
        b = torch.zeros(output_dim)

        for iteration, (X, y) in enumerate(trn_dl):
            logits = my_model(X, input_dim, w_init, b)
            # I would like to use probs to calculate losses
            # for ProSelfLC, CCE, LS, LS, LC, etc.
            probs = logits2probs_softmax(logits)
            y_vector = np.zeros((len(y), output_dim), dtype=np.float32)
            for i in range(len(y)):
                y_vector[i][y[i]] = 1
            y_vector = torch.tensor(y_vector)

            loss1 = self.loss_criterion1(probs, y_vector)
            loss2 = self.loss_criterion2(probs, y_vector)
            loss3 = self.loss_criterion3(probs, y_vector)

            print(f"loss1: {loss1.item()} " + f" versus loss2: {loss2.item()}")
            print("logits shape: " + str(logits.shape))
            print("label shape: " + str(y.shape))
            print("distribution shape: " + str(y_vector.shape))
            print("probs shape: " + str(probs.shape))

            print("gradient check")
            # logits.retain_grad()  # for intermediate variables

            # register_hook for logits
            logits.register_hook(set_grad(logits))

            loss1.backward(retain_graph=True)
            loss1_logit_grad = logits.grad

            # clear out the gradients of Variables
            # (i.e. W, b)
            # W.grad.data.zero_()
            # b.grad.data.zero_()
            # zero out so not to acculumate
            # zero out so not to acculumate
            # logits.grad.data.zero_()

            loss2.backward(retain_graph=True)
            loss2_logit_grad = logits.grad

            self.assertTrue(
                torch.all(
                    torch.lt(
                        torch.abs(torch.add(loss1_logit_grad, -loss2_logit_grad)), 1e-4
                    )
                )
            )

            loss3.backward(retain_graph=True)
            loss3_logit_grad = logits.grad
            with torch.no_grad():
                H_pred_probs = torch.sum(-(probs + 1e-12) * torch.log(probs + 1e-12), 1)
                H_pred_probs = torch.reshape(H_pred_probs, (4, 1))
                H_pred_probs = H_pred_probs.repeat(1, output_dim)
            logit_grad_derived = (
                (1 - self.params["epsilon"]) * (probs - y_vector)
                - self.params["epsilon"]
                * probs
                * (torch.log(probs + 1e-12) + H_pred_probs)
            ) / batch_size

            self.assertTrue(
                torch.all(
                    torch.lt(
                        torch.abs(torch.add(loss3_logit_grad, -logit_grad_derived)),
                        1e-4,
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
