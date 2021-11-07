import matplotlib.pyplot as plt
import numpy as np
import torch


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
    # reimplementation of F.softmax(logits)
    # or: nn.Softmax()(logits)
    # per instance:
    # subtract max logit for numerical issues
    subtractmax_logits = logits - torch.max(logits, dim=1, keepdim=True).values
    exp_logits = torch.exp(subtractmax_logits)
    sum_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    return exp_logits / sum_logits


@torch.no_grad()
def intlabel2onehot(class_num, intlabel) -> np.ndarray:
    """
    intlabel in the class index: [0, class_num-1]
    """
    target_probs = np.zeros(class_num, dtype=np.float32)
    target_probs[intlabel] = 1
    return target_probs


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
