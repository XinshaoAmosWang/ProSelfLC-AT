import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
from torch.utils.data import DataLoader

from proselflc.exceptions import ParamException
from proselflc.optim.sgd_multistep import SGDMultiStep, WarmUpLR
from proselflc.slicegetter.get_dataloader import DataLoaderPool
from proselflc.slicegetter.get_lossfunction import LossPool
from proselflc.slicegetter.get_network import NetworkPool
from proselflc.trainer.utils import logits2probs_softmax

get_network = NetworkPool.get_network
get_dataloader = DataLoaderPool.get_dataloader
get_lossfunction = LossPool.get_lossfunction
colorscale = [[0, "#4d004c"], [0.5, "#f2e5ff"], [1, "#ffffff"]]


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

        self.total_epochs = params["total_epochs"]
        # time tracker for proselflc only.
        self.loss_name = params["loss_name"]
        if self.loss_name == "proselflc":
            self.cur_time = 0
            self.counter = params["counter"]
            if self.counter == "iteration":
                # affected by batch size.
                params["total_iterations"] = self.total_epochs * len(
                    self.traindataloader
                )

        # loss function
        self.loss_criterion = get_lossfunction(params)

        # TODO: create a getter for all optional optimisers
        # optim with optimser and lr scheduler
        self.optim = SGDMultiStep(net_params=self.network.parameters(), params=params)
        self.warmup_epochs = params["warmup_epochs"]
        self.optim.warmup_scheduler = WarmUpLR(
            optimizer=self.optim.optimizer,
            total_iters=len(self.traindataloader) * self.warmup_epochs,
        )

        # logging misc ######################################
        # add summary writer
        self.summarydir = params["summary_writer_dir"]
        self.params = params
        self.noisy_data_analysis_prep()
        self.init_logger()
        # logging misc ######################################

    def noisy_data_analysis_prep(self):
        # special case for label noise
        self.cleantraindataloader = None
        sym_noisy_key = "symmetric_noise_rate"
        if sym_noisy_key in self.params.keys() and self.params[sym_noisy_key] > 0.0:
            # to get clean train data
            self.params["train"] = True
            self.noise_rate = self.params[sym_noisy_key]
            self.params[sym_noisy_key] = 0.0
            self.cleantraindataloader = get_dataloader(self.params)
            self.params[sym_noisy_key] = self.noise_rate

            mask_list = np.array(
                self.cleantraindataloader._dataset.targets
            ) == np.array(self.traindataloader._dataset.targets)

            # clean and noisy subsets
            clean_indexes = [list[0] for list in np.argwhere(mask_list)]
            noisy_indexes = [list[0] for list in np.argwhere(np.invert(mask_list))]
            assert len(noisy_indexes) == self.noise_rate * len(mask_list)
            assert len(clean_indexes) == (1 - self.noise_rate) * len(mask_list)
            clean_subset = torch.utils.data.Subset(
                self.traindataloader._dataset,
                clean_indexes,
            )
            noisy_subset = torch.utils.data.Subset(
                self.traindataloader._dataset,
                noisy_indexes,
            )
            cleaned_noisy_subset = torch.utils.data.Subset(
                self.cleantraindataloader._dataset,
                noisy_indexes,
            )
            self.clean_subloader = DataLoader(
                dataset=clean_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"],
            )
            self.noisy_subloader = DataLoader(
                dataset=noisy_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"],
            )
            self.cleaned_noisy_subloader = DataLoader(
                dataset=cleaned_noisy_subset,
                shuffle=False,
                num_workers=self.params["num_workers"],
                batch_size=self.params["batch_size"],
            )

    def init_logger(self):
        self.accuracy_dynamics = {"epoch": []}
        self.loss_dynamics = {"epoch": []}
        self.dataloaders = {}
        if self.cleantraindataloader is None:
            # clean case
            self.dataloaders = {
                "clean_train": self.traindataloader,
                "clean_test": self.testdataloader,
            }
        else:
            # noisy data
            self.dataloaders = {
                # "clean_train": self.cleantraindataloader,
                "clean_test": self.testdataloader,
                # "noisy_train": self.traindataloader,
                "noisy_subset": self.noisy_subloader,
                "clean_subset": self.clean_subloader,
                "cleaned_noisy_subset": self.cleaned_noisy_subloader,
            }
        for name in self.dataloaders.keys():
            self.accuracy_dynamics[name] = []
            self.loss_dynamics[name] = []

    def train(self) -> None:
        # #############################
        for epoch in range(1, self.total_epochs + 1):
            # train one epoch
            self.train_one_epoch(
                epoch=epoch,
                dataloader=self.traindataloader,
            )
            # lr scheduler
            if epoch > self.warmup_epochs:
                self.optim.lr_scheduler.step()

            if epoch % self.params["eval_interval"] == 0:
                print("Evaluating Network.....")
                self.loss_dynamics["epoch"].append(epoch)
                self.accuracy_dynamics["epoch"].append(epoch)
                for dataloader_name, dataloader in self.dataloaders.items():
                    (loss, accuracy) = self.evaluation(
                        dataloader=dataloader,
                    )
                    self.loss_dynamics[dataloader_name].append(loss)
                    self.accuracy_dynamics[dataloader_name].append(accuracy)

                    print(
                        dataloader_name
                        + ": Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(
                            epoch,
                            loss,
                            accuracy,
                        )
                    )

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
                    # epoch counter to iteration counter
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

            # backward
            self.optim.optimizer.zero_grad()
            if self.params["loss_name"] == "dm_exp_pi":
                # ########################################################
                # Implementation for derivative manipulation + Improved MAE
                # Novelty: From Loss Design to Derivative Design
                # Our work inspired: ICML-2020 (Normalised Loss Functions)
                # and ICML-2021 (Asymmetric Loss Functions)
                # ########################################################
                # remove orignal weights
                p_i = pred_probs[labels.nonzero(as_tuple=True)][:, None]
                logit_grad_derived = (pred_probs - labels) / (2.0 * (1.0 - p_i) + 1e-8)
                # add new weight: derivative manipulation or IMAE
                logit_grad_derived *= torch.exp(
                    self.params["dm_beta"]
                    * (1.0 - p_i)
                    * torch.pow(p_i + 1e-8, self.params["dm_lambda"])
                )
                # derivative normalisation,
                # which inspired the ICML-2020 paper-Normalised Loss Functions
                sum_weight = sum(
                    torch.exp(
                        self.params["dm_beta"]
                        * (1.0 - p_i)
                        * torch.pow(p_i + 1e-8, self.params["dm_lambda"])
                    )
                )
                logit_grad_derived /= sum_weight
                logits.backward(logit_grad_derived)
            else:
                loss.backward()

            # update params
            self.optim.optimizer.step()
            # #############################

            # warmup iteration-wise lr scheduler
            if epoch <= self.warmup_epochs:
                self.optim.warmup_scheduler.step()
                # print("epoch={}, lr={}, loss={}, bidx={}".format(
                #     epoch,
                #     self.optim.optimizer.param_groups[0]["lr"],
                #     loss.item(),
                #     batch_index,
                # ))

    @torch.no_grad()
    def evaluation(self, dataloader):
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

        return test_loss, test_accuracy

    def sink_csv_figures(self):
        # logging misc ######################################
        accuracy_dynamics_df = pd.DataFrame(self.accuracy_dynamics)
        loss_dynamics_df = pd.DataFrame(self.loss_dynamics)
        tosink_dataframes = {
            "accuracy": accuracy_dynamics_df,
            "loss": loss_dynamics_df,
        }
        file_name = "_".join(tosink_dataframes.keys())
        ########################
        xlsx_writer = pd.ExcelWriter(
            "{}/{}.xlsx".format(self.summarydir, file_name), engine="xlsxwriter"
        )
        for dfname, dfdata in tosink_dataframes.items():
            dfdata.to_excel(xlsx_writer, sheet_name=dfname)
        xlsx_writer.close()

        ########################
        html_writer = open("{}/{}.html".format(self.summarydir, file_name), "w")
        for dfname, dfdata in tosink_dataframes.items():
            fig = ff.create_table(
                dfdata,
                index=True,
                colorscale=colorscale,
            )
            # Make text size larger
            for i in range(len(fig.layout.annotations)):
                fig.layout.annotations[i].font.size = 12

            html_writer.write(
                "<p style="
                + "text-align:center"
                + ">{}</p>".format(dfname)
                + fig.to_html()
                + "<p>&nbsp;&nbsp;</p>"
            )
        html_writer.close()

        ########################
        # save figures
        for dfname, dfdata in tosink_dataframes.items():
            dfdata.index = dfdata["epoch"]
            dfdata = dfdata.drop(columns=["epoch"])
            dfdata.plot.line()
            plt.savefig(
                "{}/{}.pdf".format(self.summarydir, dfname),
                dpi=100,
            )
