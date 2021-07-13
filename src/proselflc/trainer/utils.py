import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def logits2probs_softmax(logits):
    """
    Transform logits to probabilities using exp function and normalisation

    Input:
        logits with shape: (N, C)
        N means the batch size or the number of instances.
        C means the number of training classes.

    Output:
        probability vectors of shape (N, C)
    """
    exp_logits = torch.exp(logits)
    sum_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    return exp_logits / sum_logits


def intlabel2onehot(device: str, class_num, intlabels) -> Tensor:
    intlabels = intlabels.cpu()
    target_probs = np.zeros((len(intlabels), class_num), dtype=np.float32)
    for i in range(len(intlabels)):
        target_probs[i][intlabels[i]] = 1
    target_probs = torch.from_numpy(target_probs)
    if device == "gpu":
        target_probs = target_probs.cuda()
        intlabels = intlabels.cuda()
    return target_probs


# def intlabel2onehot(class_num, intlabels) -> Tensor:
#     target_probs = np.zeros(
#         (len(intlabels), class_num),
#         dtype=np.float32
#     )
#     for i in range(len(intlabels)):
#         target_probs[i][intlabels[i]] = 1
#     target_probs = torch.tensor(
#         target_probs,
#         dtype=torch.float32,
#         device=intlabels.device,
#     )
#     return target_probs


def save_figures(fig_save_path="", y_inputs=[], fig_legends=[], xlabel="", ylabel=""):
    colors = ["r", "b"]
    linestyles = ["solid", "dashdot"]
    x = torch.arange(len(y_inputs[0]))
    #
    fig, ax = plt.subplots()
    for y_input, color, linestyle, fig_legend in zip(
        y_inputs, colors, linestyles, fig_legends
    ):
        ax.plot(x, y_input, color=color, linestyle=linestyle, label=fig_legend)
    # legend = ax.legend(loc="upper right")
    ax.legend(loc="upper right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #
    plt.savefig(fig_save_path, dpi=100)
