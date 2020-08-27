import re
import json
import os

import torch
import numpy as np
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

from ..transforms import Compose, TensorResize
from ..transforms.numpy_transforms import Resize
from . import losses

__all__ = ['BaseDict', 'ClassDict', 'Processor', 'get_transforms', 'default_collate']


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
        "detach": lambda x: x.detach(),
        "to_numpy": lambda x: x.detach().cpu().numpy(),
        "to_tensor": torch.from_numpy,
        "flatten": lambda x: x.flatten(),
        "to_cpu": lambda x: x.cpu(),
        "tolist": lambda x: x.tolist(),
        "resize224": TensorResize(224),

    })
    return TRANSFORMS


def get_video_io_transforms():
    TRANSFORMS = Processor({
        "copy": lambda x: x.copy(),
        "resize224": Resize((224, 224)),
        "resize112": Resize((112, 112))
    })
    return TRANSFORMS


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    else:
        return batch
    raise TypeError(default_collate_err_msg_format.format(elem_type))
