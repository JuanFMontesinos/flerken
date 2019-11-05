#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..dataloaders.transforms import Compose

from . import losses
import torch
import numpy as np

import json
import os

__all__ = ['BaseDict','ClassDict', 'Processor', 'get_transforms']


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

class BaseDict(dict):
    def __add__(self, other):
        o_keys = other.keys()
        for key in self.keys():
            if key in o_keys:
                raise KeyError('Cannot concatenate both dictionaries. Key %s duplicated' % key)
        self.update(other)
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def write(self, path):
        path = os.path.splitext(path)[0]
        with open('%s.json' % path, 'w') as outfile:
            json.dump(self, outfile)

    def load(self, path):
        with open(path, 'r') as f:
            datastore = json.load(f)
            self.update(datastore)
        return self
class MutableList(list):
    pass

class ClassDict(BaseDict):
    """
    Object like dict, every dict[key] can be visited by dict.key
    """

    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, key, value):
        self.update({key: value})

class Processor(BaseDict):
    def __call__(self, *args):
        return Compose([self[x] for x in args])


def get_transforms():
    TRANSFORMS = Processor({
        "float": lambda x: x.float() if torch.is_tensor(x) else x.astype(np.float32),
        "copy": lambda x: x.clone() if torch.is_tensor(x) else x.copy(),
        "detach": lambda x:x.detach(),
        "to_numpy": lambda x: x.detach().cpu().numpy(),
        "to_tensor": torch.from_numpy,
        "flatten": lambda x:x.flatten(),
        "to_cpu": lambda x:x.cpu()

    })
    return TRANSFORMS
