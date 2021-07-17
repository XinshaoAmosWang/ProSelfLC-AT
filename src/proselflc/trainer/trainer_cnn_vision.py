import pandas as pd
import torch

from proselflc.exceptions import ParamException
from proselflc.optim.sgd_multistep import SGDMultiStep
from proselflc.slicegetter.get_dataloader import DataLoaderPool
from proselflc.slicegetter.get_lossfunction import LossPool
from proselflc.slicegetter.get_network import NetworkPool
from proselflc.trainer.utils import logits2probs_softmax, save_figures

get_network = NetworkPool.get_network
get_dataloader = DataLoaderPool.get_dataloader
get_lossfunction = LossPool.get_lossfunction


class Trainer:
    """

    Inputs:
        1. dataloader
        2. network with train mode: network.train(mode=True)
        3. loss
        4. optimiser
        5. device = cpu or gpu

    Functionality:
        1. build the graph according to params
            dataloader,
            network,
            loss function,
            optimiser
        2. batch training through dataloader, which is iterable.
        3.
    """

    def __init__(self, params):
        if "device" not in params.keys() or params["device"] not in ["cpu", "gpu"]:
            error_msg = (
                "The input params have no key of device. "
                + "params["
                + "device"
                + "] "
                + " has to be provided as cpu or gpu."
            )
            raise (ParamException(error_msg))
        self.device = params["device"]

        # network
        self.network_name = params["network_name"]
        self.network = get_network(params)
        self.network.train(mode=True)
        if self.device == "gpu":
            self.network = self.network.cuda()

        # dataloader
        params["train"] = True
        self.traindataloader = get_dataloader(params)
        params["train"] = False
        self.testdataloader = get_dataloader(params)
        self.data_name = params["data_name"]

        # loss function
        self.loss_name = params["loss_name"]
        self.loss_criterion = get_lossfunction(params)

        # TODO: create a getter for all optional optimisers
        # optim with optimser and lr scheduler
        self.optim = SGDMultiStep(net_params=self.network.parameters(), params=params)

        self.total_time = params["total_time"]
        # time tracker for proselflc only.
        if self.loss_name == "proselflc":
            self.cur_time = 0
            self.counter = params["counter"]

        # logging misc ######################################
        # add summary writer
        self.summarydir = params["summary_writer_dir"]
        self.params = params
        self.init_logger()
        # logging misc ######################################

    def train(self) -> None:
        if self.loss_name == "proselflc" and self.counter == "iteration":
            # to epoch
            print(len(self.traindataloader))
            self.total_time = self.total_time / len(self.traindataloader)
            self.total_time = int(self.total_time)
            if self.total_time < 2:
                error_msg = (
                    "self.total_time = "
                    + str(self.total_time)
                    + ", is too small. Please check settings of self.counter and"
                    + "self.total_time. "
                )
                raise ParamException(error_msg)

        # #############################
        for epoch in range(1, self.total_time + 1):
            # train one epoch
            self.train_one_epoch(
                epoch=epoch,
                dataloader=self.traindataloader,
            )
            # evaluation one epoch
            (loss, accuracy) = self.evaluation(
                epoch=epoch,
                dataloader=self.traindataloader,
                data_usagename="traindata",
            )
            self.accuracy_dynamics["train"].append(accuracy)
            self.loss_dynamics["train"].append(loss)

            (loss, accuracy) = self.evaluation(
                epoch=epoch,
                dataloader=self.testdataloader,
                data_usagename="testdata",
            )

            self.accuracy_dynamics["test"].append(accuracy)
            self.loss_dynamics["test"].append(loss)

            # lr scheduler
            self.optim.lr_scheduler.step()
        # #############################

        self.sink_csv_figures()

    def train_one_epoch(self, epoch: int, dataloader) -> None:
        self.network.train()  # self.network.train(mode=True)

        for batch_index, (raw_inputs, labels) in enumerate(dataloader):
            # #############################
            # track time for proselflc
            if self.loss_name == "proselflc":
                if self.counter == "epoch":
                    self.cur_time = epoch
                else:
                    self.cur_time = (epoch - 1) * len(dataloader) + batch_index + 1
            # #############################

            # #############################
            # data ingestion
            network_inputs = raw_inputs
            if self.device == "gpu":
                network_inputs = network_inputs.cuda()
                labels = labels.cuda()
            # #############################

            # #############################
            # forward
            logits = self.network(network_inputs)
            pred_probs = logits2probs_softmax(logits=logits)
            # #############################

            # #############################
            # loss
            if self.loss_name != "proselflc":
                loss = self.loss_criterion(
                    pred_probs=pred_probs,
                    target_probs=labels,
                )
            else:
                loss = self.loss_criterion(
                    pred_probs=pred_probs,
                    target_probs=labels,
                    cur_time=self.cur_time,
                )
            # #############################

            # #############################
            # backward
            self.optim.optimizer.zero_grad()
            loss.backward()
            # update params
            self.optim.optimizer.step()
            # #############################

    @torch.no_grad()
    def evaluation(self, epoch, dataloader, data_usagename: str):
        self.network.eval()

        test_loss = 0.0
        test_correct = 0.0

        for iter_idx, (raw_inputs, labels) in enumerate(dataloader):
            # #############################
            # data ingestion
            network_inputs = raw_inputs
            if self.device == "gpu":
                network_inputs = network_inputs.cuda()
                labels = labels.cuda()
            # #############################

            # #############################
            # forward
            logits = self.network(network_inputs)
            pred_probs = logits2probs_softmax(logits=logits)
            # #############################

            # #############################
            # loss
            if self.loss_name != "proselflc":
                loss = self.loss_criterion(
                    pred_probs=pred_probs,
                    target_probs=labels,
                )
            else:
                loss = self.loss_criterion(
                    pred_probs=pred_probs,
                    target_probs=labels,
                    cur_time=self.cur_time,
                )
            # #############################

            test_loss += loss.item()
            _, preds = pred_probs.max(1)
            _, annotations = labels.max(1)
            test_correct += preds.eq(annotations).sum()

        test_loss = test_loss / len(dataloader.dataset)
        test_accuracy = test_correct.item() / len(dataloader.dataset)
        print("Evaluating Network.....")
        print(
            data_usagename
            + ": Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}".format(
                epoch,
                test_loss,
                test_accuracy,
            )
        )
        self.txtwriter = open(
            self.summarydir + "/learning_dynamics.txt",
            "a",
        )
        self.txtwriter.write(
            "epoch="
            + str(epoch)
            + ", "
            + data_usagename
            + ": "
            + "loss="
            + str(test_loss)
            + ", "
            + "accuracy="
            + str(test_accuracy)
            + "\n"
        )
        self.txtwriter.close()

        return test_loss, test_accuracy

    def init_logger(self):
        self.accuracy_dynamics = {
            "train": [],
            "test": [],
        }
        self.loss_dynamics = {
            "train": [],
            "test": [],
        }

    def sink_csv_figures(self):
        # logging misc ######################################
        # train finished: save figures
        fig_save_path = self.summarydir + "/accuracy_dynamics.pdf"
        y_inputs = [self.accuracy_dynamics["train"], self.accuracy_dynamics["test"]]
        fig_legends = ["train accuracy", "test accuracy"]
        fig_xlabel = "Epoch"
        fig_ylabel = "Accuracy"
        save_figures(fig_save_path, y_inputs, fig_legends, fig_xlabel, fig_ylabel)
        df = pd.DataFrame(
            {
                fig_legends[0]: y_inputs[0],
                fig_legends[1]: y_inputs[1],
            }
        )
        df.to_csv(
            self.summarydir + "/accuracy_dynamics.csv", encoding="utf-8", index=False
        )
        #
        fig_save_path = self.summarydir + "/loss_dynamics.pdf"
        y_inputs = [self.loss_dynamics["train"], self.loss_dynamics["test"]]
        fig_legends = ["train loss", "test loss"]
        fig_ylabel = "Loss"
        save_figures(fig_save_path, y_inputs, fig_legends, fig_xlabel, fig_ylabel)
        df = pd.DataFrame(
            {
                fig_legends[0]: y_inputs[0],
                fig_legends[1]: y_inputs[1],
            }
        )
        df.to_csv(self.summarydir + "/loss_dynamics.csv", encoding="utf-8", index=False)
        # logging misc ######################################
