import torch
import numbers
import numpy as np
from collections.abc import Iterable
from six import callable
import time
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import cycle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, hist=False):
        self.track_hist = hist
        self.reset()
        self.called = False

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.track_hist:
            self.hist = []

    def update(self, val, n=1):
        self.called = True
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.track_hist:
            self.hist.append(val)


class TensorAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.called = False

    def reset(self):
        self.count = 0
        self.hist = torch.Tensor()

    def update(self, val):
        assert isinstance(val, torch.Tensor)
        self.called = True
        self.count += val.nelement()
        self.hist = torch.cat((self.hist, self.tensor2data(val).float().flatten()))

    def __iter__(self):
        return iter(self.hist)

    @staticmethod
    def tensor2data(tensor):
        if tensor.nelement() == 0:
            raise Exception('Empty object cannot be attached to tensorboard')
        else:
            return tensor.detach().cpu()

    def numpy(self):
        return self.hist.numpy()


class AccMeter(object):

    def __init__(self, labels):
        self.gt = TensorAverageMeter()
        self.pred = TensorAverageMeter()
        self.epoch_array = AverageMeter(hist=True)

        self.cmflag = True if labels is not None else False
        self.labels = labels
        self.t2l = {}
        if self.cmflag:
            for idx, label in enumerate(labels):
                self.t2l.update({idx: label})

    def __call__(self, mode):
        if mode == 'gt':
            return self.gt
        elif mode == 'pred':
            return self.pred
        else:
            raise TypeError

    def update(self, mode, array):
        if mode == 'gt':
            self.gt.update(array)
        elif mode == 'pred':
            self.pred.update(array.round())
        else:
            raise TypeError

    def tensor2label(self, tensor):
        return [self.t2l[int(idx.item())] for idx in tensor]

    def update_epoch(self):
        acc = accuracy_score(self.gt.numpy(), self.pred.numpy())
        self.epoch_array.update(acc)
        if self.cmflag:
            self.confusion_matrix = confusion_matrix(self.tensor2label(self.gt),
                                                     self.tensor2label(self.pred),
                                                     self.labels)
        self.gt.reset()
        self.pred.reset()
        return self.epoch_array.val

    def plot_confusion_matrix(self, normalize, title, cmap):
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
        cm = self.confusion_matrix
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


class AccuracyItem(object):
    def __init__(self, labels):
        self.tuple = {'train': AccMeter(labels), 'val': AccMeter(labels), 'test': AccMeter(labels)}
        self.enabled = 0x110101

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, mask):
        mask = '0x{:06x}'.format(mask)[2:]  # Converts mask to string and removes 0x
        self.tuple['train'].gt.enabled = self.tuple['train'].pred.enabled = bool(int(mask[0]))
        self.tuple['train'].epoch_array.enabled = bool(int(mask[1]))
        self.tuple['val'].gt.enabled = self.tuple['val'].pred.enabled = bool(int(mask[2]))
        self.tuple['val'].epoch_array.enabled = bool(int(mask[3]))
        self.tuple['test'].gt.enabled = self.tuple['test'].pred.enabled = bool(int(mask[4]))
        self.tuple['test'].epoch_array.enabled = bool(int(mask[5]))
        self._enabled = mask

    def update(self, mode, gt, pred):
        self.tuple[mode].update('gt', gt)
        self.tuple[mode].update('pred', pred)

    def update_epoch(self, mode):
        return self.tuple[mode].update_epoch()

    def plot_confusion_matrix(self, mode, normalize=False, title=None, cmap=plt.cm.BuGn):
        return self.tuple[mode].plot_confusion_matrix(normalize, title, cmap)

    def get_acc(self, mode):
        return self.tuple[mode].epoch_array.val


class NullItem(object):
    def __init__(self, *args, **kwargs):
        pass

    def constructor(self, *args, **kwargs):
        pass

    def update_epoch(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def plot_confusion_matrix(self, *args, **kwargs):
        return None

    def get_acc(self, *args, **kwargs):
        return None


class TensorAccuracyItem(AccuracyItem):
    def __init__(self, labels, obj=None, gttransforms=None, predtransforms=None):
        super(TensorAccuracyItem, self).__init__(labels)
        self.constructor(obj)
        self._valid_states = ('val', 'test', 'train')
        self.gt_transforms = gttransforms
        self.pred_transforms=predtransforms

    def constructor(self, obj):
        if obj is not None:
            self.__dict__.update(obj.__dict__)
            # TODO This way end will mismatch as its initializations has been overwritten

    def __call__(self, mode, gt, pred):
        assert mode in self._valid_states
        gt = self.tensor2data(gt)
        pred = self.tensor2data(pred)
        if self.gt_transforms is not None:
            gt = self.gt_transforms(gt)
        if self.pred_transforms is not None:
            pred = self.pred_transforms(pred)

        self.update(mode, gt, pred)

    @staticmethod
    def tensor2data(tensor):
        if tensor.nelement() == 0:
            raise Exception('Empty object cannot be attached to tensorboard')
        else:
            return tensor.detach().cpu()


class Array(object):
    def __init__(self):
        self.array = AverageMeter(hist=True)
        self.epoch_array = AverageMeter(hist=True)

    def update(self, value):
        self.array.update(value)

    def update_epoch(self):
        value = np.mean(self.array.hist)
        self.epoch_array.update(value)
        self.array.reset()
        return value


class Timers(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.tuple = {'train': Array(), 'val': Array(), 'test': Array()}
        self.end = time.time()
        self.enabled = 0x110101

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, mask):
        mask = '0x{:06x}'.format(mask)[2:]  # Converts mask to string and removes 0x
        self.tuple['train'].array.enabled = bool(int(mask[0]))
        self.tuple['train'].epoch_array.enabled = bool(int(mask[1]))
        self.tuple['val'].array.enabled = bool(int(mask[2]))
        self.tuple['val'].epoch_array.enabled = bool(int(mask[3]))
        self.tuple['test'].array.enabled = bool(int(mask[4]))
        self.tuple['test'].epoch_array.enabled = bool(int(mask[5]))
        self._enabled = mask

    def update(self, mode, value):
        self.tuple[mode].update(value)

    def update_timed(self):
        self.data_time.update(time.time() - self.end)

    def update_timeb(self):
        self.batch_time.update(time.time() - self.end)

    def update_epoch(self, mode):
        return self.tuple[mode].update_epoch()

    def print_logger(self, t, j, iterations, logger):
        logger.info('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            t, j, iterations, batch_time=self.batch_time,
            data_time=self.data_time, loss=self.tuple['train'].array))

    def epoch_logger(self, current_epoch, total_epochs, epoch_loss, logger):
        logger.info('Epoch: [{0}/{1}]\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss ({loss:.4f})'.format(
            current_epoch, total_epochs, data_time=self.data_time, loss=epoch_loss))

    def is_best(self):
        if self.tuple['val'].epoch_array.called:
            mode = 'val'
        else:
            mode = 'train'

        return (np.mean(self.tuple[mode].array.hist) <= np.min(self.tuple[mode].epoch_array.hist))


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TensorScalarItem(object):
    def __init__(self, obj=None):
        self.constructor(obj)
        self._valid_states = ('val', 'test', 'train')

    def constructor(self, obj=None):
        self.data = Timers()
        if obj is not None:
            self.data.__dict__.update(obj.__dict__)  # TODO shouldn't be self.__dict__?
            # TODO This way end will mismatch as its initializations has been overwritten

    def update_epoch(self, mode):
        return self.data.tuple[mode].update_epoch()

    def __call__(self, _scalar, mode):
        assert _scalar.nelement() == 1
        assert mode in self._valid_states

        self.scalar = _scalar.item()
        self.data.update_timeb()
        self.data.end = time.time()
        self.data.tuple[mode].update(self.scalar)
        return self.scalar


class TensorBItem(object):
    def __init__(self, **config):
        self.initialized = False
        self._keywords = ('function', 'transforms', 'freq', 'tag', 'main_tag')
        self.config = self.gather_config(config)
        self.auto = not bool(
            self.config)  # TODO Comparison between dict and boolean without bool() ---> ERROR look for it
        if 'transforms' in config.keys():  # Check if transforms has been gathered from config
            self.transforms = self.set_transforms(self.transforms)
        else:
            self.transforms = lambda x: x  # Null transformation

    def constructor(self, item):
        self.initialized = True
        self.get_properties(item)

        if self.is_tensor():
            self.is_torch_number, item = self.tensor2data(item)
            # Update properties in case tensor became numerical
        self.get_properties(item)

    def gather_config(self, config):
        for key in config.keys():
            if key in self._keywords:
                setattr(self, key, config.pop(key))
        return config

    def get_properties(self, item):
        self.type = self.get_type(item)
        self.isiterable = isinstance(item, Iterable)
        return True

    def is_tensor(self):
        return isinstance(self.type, torch.Tensor)

    def is_torch_number(self):
        return self.is_torch_number

    def is_numpy(self):
        return isinstance(self.type, np.ndarray)

    def is_number(self):
        return isinstance(self.type, numbers.Number)

    def is_tuple_list(self):
        return isinstance(self.type, (tuple, list))

    def is_str(self):
        return isinstance(self.type, str)

    def is_scalar(self):
        return self.is_torch_number or self.is_number()

    @staticmethod
    def set_transforms(transforms):
        assert callable(transforms)
        return transforms

    @staticmethod
    def get_type(item):
        if isinstance(item, numbers.Number):
            return numbers.Number
        elif isinstance(item, (tuple, list)):
            return type(item)
        elif isinstance(item, torch.Tensor):
            return torch.Tensor
        elif isinstance(item, np.ndarray):
            return np.ndarray
        elif isinstance(item, str):
            return str
        else:
            raise TypeError('Not implemented type')

    @staticmethod
    def tensor2data(tensor):
        if tensor.nelement() == 1:
            return True, tensor.item()
        elif tensor.nelement() == 0:
            raise Exception('Empty object cannot be attached to tensorboard')
        else:
            return False, tensor.detach().cpu()

    @staticmethod
    def ctx_tag(state, name):
        if state == 'train':
            return name + '_' + state

    def _call_scalars(self, item, parent):
        if self.is_torch_number():
            assert item.nelement() == 1
            item = self.tensor2data(item)

        function = getattr(parent.writer, self.function)
        local_cfg = self.config
        local_cfg.update({'global_step': parent.absolute_iter})
        if self.function == 'add_scalars':
            function(self.main_tag, item, **local_cfg)
        else:
            function(self.tag, item, **local_cfg)

    def _call(self, item, parent):
        if self.is_scalar():
            self._call_scalars(item, parent)
        else:
            raise NotImplementedError

    def __call__(self, item, parent):
        if not self.initialized:
            self.constructor(item)

        return self._call(self, item, parent)


class NullScheduler(_LRScheduler):
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __repr__(self):
        return 'Empty Scheduler'


def isscheduler(x):
    return isinstance(x, _LRScheduler) or isinstance(x, ReduceLROnPlateau)


class Scheduler(object):
    def __new__(cls, obj):
        def __repr__(self):
            array = self.__class__.__name__ + '(\n'
            dic = self.state_dict()
            for key in dic:
                array += '\t {0}: {1} \n'.format(key, dic[key])
            array += ')\n'
            return array

        def show_lr(self):
            lr = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr.append('{:.4e}'.format(float(param_group['lr'])))
            return lr

        def state_dict(self):
            dic = self.__class__.state_dict(self)
            dic.pop('show_lr')
            dic.pop('state_dict')
            return dic

        if not isscheduler(obj):
            obj = NullScheduler()
        else:
            import types
            obj.__class__.__repr__ = __repr__
            obj.state_dict = types.MethodType(state_dict, obj)
            obj.show_lr = types.MethodType(show_lr, obj)
        return obj


class GradientPlotter(object):  # TODO Temporal class in dev
    def __init__(self, freq, *args, **kwargs):
        self.c = cycle(iter(range(freq)))
        self.freq = freq

    def __call__(self, named_parameters):
        f = next(self.c)
        if f == self.freq - 1:
            return self.plot_gradients(named_parameters)
        else:
            return None

    @staticmethod
    def plot_gradients(named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        fig = plt.figure()
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                if p.grad is not None:
                    val_mean = p.grad.abs().mean()
                    val_max = p.grad.abs().max()
                else:
                    val_mean = val_max = 0
                ave_grads.append(val_mean)
                max_grads.append(val_max)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        return fig
