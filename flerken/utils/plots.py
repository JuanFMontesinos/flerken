from sklearn.metrics import auc
import matplotlib.pyplot as plt
import torch
from six import callable
from collections.abc import Iterable


def plot_roc(fpr, tpr, ths):
    assert len(fpr) == len(tpr) == len(ths)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    ax2.plot(ths, tpr)
    ax2.set_xlabel(r"Threshold")
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=1,
             label='ROC fold (AUC = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver operating characteristic')
    ax1.legend(loc="lower right")
    plt.tight_layout()
    return fig


def roc_curve(loader, ths, function):
    assert callable(function)
    assert isinstance(loader, Iterable)
    fpr_list = []
    tpr_list = []

    if isinstance(ths, list):
        pass
    elif isinstance(ths, tuple):
        assert ths[0] < ths[1]
        ths = torch.linspace(ths[0], ths[1], steps=ths[2])
    for th in ths:
        fpr_rate, tpr_rate = function(loader, th)
        fpr_list.append(fpr_rate)
        tpr_list.append(tpr_rate)

    return (fpr_list, tpr_list)
