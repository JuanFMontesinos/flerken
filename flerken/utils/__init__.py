#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import losses
import torch


def classification_metric(pred_labels, true_labels):
    pred_labels = torch.ByteTensor(pred_labels)
    true_labels = torch.ByteTensor(true_labels)

    assert 1 >= pred_labels.all() >= 0
    assert 1 >= true_labels.all() >= 0

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = torch.sum((pred_labels == 1) & ((true_labels == 1)))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = torch.sum((pred_labels == 0) & (true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = torch.sum((pred_labels == 1) & (true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = torch.sum((pred_labels == 0) & (true_labels == 1))
    return (TP, TN, FP, FN)
