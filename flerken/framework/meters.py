from ..utils import BaseDict, MutableList, Processor, get_transforms, Compose
from functools import partial
from six import callable
import numpy as np
import torch
from functools import wraps
from warnings import warn
from torch.nn import BCEWithLogitsLoss, BCELoss, NLLLoss, CrossEntropyLoss
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


class AccFunction(object):

    def __init__(self, labels):

        self.cmflag = True if labels is not None else False
        self.labels = labels
        self.t2l = {}
        if self.cmflag:
            for idx, label in enumerate(labels):
                self.t2l.update({idx: label})

    def __call__(self, gt, pred):
        acc = accuracy_score(gt.numpy(), pred.numpy())
        if self.cmflag:
            cm = confusion_matrix(self.tensor2label(gt),
                                  self.tensor2label(pred),
                                  self.labels)
            return [acc, cm]
        return [acc, None]

    def tensor2label(self, tensor):
        return [self.t2l[int(idx.item())] for idx in tensor]

    def plot_confusion_matrix(self, cm, normalize, title, cmap):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        # Only use the labels that appear in the data
        classes = self.labels
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return [fig, ax]


class TensorHandler(object):
    def __init__(self, *args, **kwargs):
        TRANSFORMS = get_transforms()
        base_processor = Processor(TRANSFORMS)
        custom_procesor = Processor(kwargs)
        self.processor = (base_processor + custom_procesor)(*args)

    def __add__(self, other):
        self.processor = self.processor + other.processor
        return self

    def __call__(self, tensor):
        return self.processor(tensor)

    def __str__(self):
        return self.processor.__str__()

    def __repr__(self):
        return self.processor.__repr__()


class TensorStorage(BaseDict):
    def __init__(self, handlers, opt, on_the_fly=False, redirect=None, *args, **kwargs):
        super(TensorStorage, self).__init__(*args, **kwargs)
        self.handlers = handlers
        self.opt = opt  # {'key':{'type':'input','store':'list'},'key2':{'type':'output','store':'tensor'}}
        self.outputs = [x for x in self.opt if self.opt[x]['type'] == 'output']
        for output in self.outputs:
            self[output] = []
        self.on_the_fly = on_the_fly
        self.redirect = redirect
        self.init()

    def __getitem__(self, key):
        if key not in self and key in self.outputs:
            self[key] = list()
        return super(TensorStorage, self).__getitem__(key)

    def send(self, key, value):
        value_processed = self.handlers[key](value)
        value_to_set = self.store_opt(key, value_processed)
        # super(TensorStorage, self).__setitem__(key, value_to_set)
        self[key] = value_to_set
        return value_to_set

    def reset(self):
        for key in list(self.keys()):
            if key not in self.outputs:
                del self[key]
        self.init()

    def init(self):
        for key in self.opt.keys():
            if self.opt[key]['store'] == 'list':
                self[key] = list()

    def store_opt(self, key, value):
        if self.opt[key]['store'] == 'list':
            return self._store_opt_list(key, value)
        elif self.opt[key]['store'] == 'stack':
            return self._store_opt_stack(key, value)
        else:
            raise KeyError('Storage options are whether list or stack but %s given' % self.opt[key])

    def process(self, *args, store=False):
        iterable = zip(*[self[key] for key in args])
        for inputs in iterable:
            results = {}
            for output in self.outputs:
                result = self.handlers[output](*inputs)
                results[output] = result
            if store:
                self[output].append(result)
        self.reset()

        return results

    def _store_opt_list(self, key, value):
        out = self.get(key)
        if out is None:
            out = []
        out.append(value)
        return out

    def _store_opt_stack(self, key, value):
        if self.get(key) is None:
            return value[None, ...]
        if torch.is_tensor(value):
            return torch.cat([self[key], value[None, ...]])
        elif isinstance(value, np.ndarray):
            return np.concatenate([self[key], value[None, ...]])
        else:
            raise TypeError('TensorStorage only support np.ndarray or torch.tensor for stacking mode')


class ScalarListMeter(BaseDict):
    """Computes and stores the average and current value"""

    def __init__(self, hist=True, function=np.mean, is_best=[max, lambda epoch, optimal: optimal > epoch]):
        self.track_hist = hist
        self.reset()
        self.reset_epoch()
        self.called = False
        self.f = function

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track_hist:
            self['iter'] = MutableList()
            self['iter'].enabled = False
    def reset_epoch(self):
        self['epoch'] = MutableList()
        self['epoch'].enabled = True

    def send(self, val, n=1):
        self.called = True
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.track_hist:
            self['iter'].append(val)

    def process(self, store=False):
        value = self.f(self['iter'])
        self.reset()
        if store:
            self['epoch'].append(value)
        return value

    def is_best(self, criteria):
        if not bool(self['epoch'][:-1]):
            return True
        else:
            optimal = criteria[0](self['epoch'][:-1])
            epoch = self['epoch'][-1]
            return criteria[1]( epoch,optimal)


class Meter(BaseDict):
    def __init__(self, obj, default=None, *args, **kwargs):
        super(Meter, self).__init__(*args, **kwargs)
        if not callable(obj):
            raise RuntimeError('Meter`s input obj should be callable, not an instance of an object')
        self.obj = obj
        if default is not None:
            if isinstance(default, (tuple, list)):
                for mode in default:
                    self.mode_constructor(mode)
            else:
                raise TypeError('Argument default should be list or tuple of strings but %s got' % str(type(default)))
        self._enabled = True

    def mode_constructor(self, mode):
        self[mode] = self.obj()

    def __getitem__(self, key):
        if key not in self:
            self.mode_constructor(key)
        return super(Meter, self).__getitem__(key)

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, mode):
        assert mode == True or mode == False
        self._enabled = mode


def get_scalar_warehouse(**kwargs):
    return get_nested_meter(partial(ScalarListMeter, **kwargs), 1)


def get_acc_meter(labels, loss, on_the_fly):
    TRANSFORM = {torch.nn.BCEWithLogitsLoss.__name__: Compose([torch.sigmoid, lambda x: (x > 0.5).float()]),
                 torch.nn.BCELoss.__name__: Compose([lambda x: (x > 0.5).float()]),
                 torch.nn.NLLLoss.__name__: Compose([lambda x: (torch.argmax(torch.softmax(x, 1), dim=1)).float()]),
                 torch.nn.CrossEntropyLoss.__name__: Compose([lambda x: (torch.argmax(x, dim=1)).float()])}

    handlers = {}
    handlers['gt'] = TensorHandler('detach', 'to_cpu', 'flatten')
    handlers['pred'] = TensorHandler('detach', 'to_cpu', 'flatten', loss.__class__.__name__, **TRANSFORM)
    handlers['epoch'] = AccFunction(labels)
    opt = {'gt': {'type': 'input', 'store': 'stack'},
           'pred': {'type': 'input', 'store': 'stack'},
           'epoch': {'type': 'output', 'store': 'list'}
           }
    return get_nested_meter(
        partial(TensorStorage, handlers=handlers, opt=opt, on_the_fly=on_the_fly, redirect={'epoch': 'acc'}), 1)


def get_loss_meter(loss):
    handlers = {}
    handlers['gt'] = lambda x: x
    handlers['pred'] = lambda x: x
    handlers['vs'] = lambda x: x # Revisar
    handlers['iter'] = loss
    opt = {'gt': {'type': 'input', 'store': 'list'},
           'pred': {'type': 'input', 'store': 'list'},
           'vs': {'type': 'input', 'store': 'list'},
           'iter': {'type': 'output', 'store': 'list'}
           }
    return get_nested_meter(
        partial(TensorStorage, handlers=handlers, opt=opt, on_the_fly=True, redirect={'iter': 'loss'}), 1)


def get_nested_meter(obj, N, default=None):
    if isinstance(default, (tuple, list)):
        assert len(default) == N
    elif default is None:
        pass
    else:
        raise TypeError('Argument default from get_nested_meter should be a list of lists of default modes')
    iterable = [None for _ in range(N)] if default is None else default

    for i, mode_list in enumerate(iterable[::-1]):
        if i == 0:
            meter = partial(Meter, obj=obj, default=mode_list)
        elif i == N - 1:
            meter = Meter(obj=meter, default=mode_list)
        else:
            meter = partial(Meter, obj=meter, default=mode_list)
    return meter()
