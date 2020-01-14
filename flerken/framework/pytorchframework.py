#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import uuid
import numpy as np
import logging
import random
import sys
import shutil
import datetime
from collections import OrderedDict
import subprocess
from . import utils as ptutils, network_initialization, sqlite_tools
from .traceback_gradients import tracegrad
from . import classitems
from .classitems import GradientPlotter as tracegrad
from ..utils import BaseDict
from . import *
from tqdm import tqdm
from functools import wraps
from ._options import *
from .debug import NaNError, InfError, Debugger
from . import meters

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARDX = False
except:
    from tensorboardX import SummaryWriter

    TENSORBOARDX = True

from functools import partial

LOGGING_FORMAT_B = "[%(filename)s: %(funcName)s] %(message)s]"
LOGGIN_FORMAT_A = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"

__all__ = ['config', 'assert_workdir', 'pytorchfw']


def set_training(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        self.hyperparameters()
        self.__atribute_assertion__()

        if not hasattr(self, 'init_function'):
            self.init_function = partial(network_initialization.init_weights, init_type=self.initializer)
        self.scheduler = classitems.Scheduler(self.scheduler)
        self.init_loss()
        self._train()

        self.key['LR'] = self.LR  # TODO Learning rate reduction by hand does not overwrite optimizer
        self.key['OPTIMIZER'] = str(self.optimizer)
        self.key['MODEL'] = self.model_version
        self.key['SCHEDULER'] = str(self.scheduler)
        self.key['BATCH_SIZE'] = int(self.batch_size)
        self.__update_db__()
        return func(*args, **kwargs)

    return inner


def config(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        self.set_config()

        return func(*args, **kwargs)

    return inner


def checkpoint_on_key(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        self.key['CHECKPOINT'] = 1
        self.__update_db__()
        return func(*args, **kwargs)

    return inner


def assert_workdir(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        assert self.workdir_enabled
        return func(*args, **kwargs)

    return inner


def dl():
    return torch.rand(2), [torch.rand(5), torch.rand(5)]


def create_folder(path):
    if not os.path.exists(path):
        os.umask(0)  # To mask the permission restrictions on new files/directories being create
        os.makedirs(path, 0o755)  # setting permissions for the folder


class framework(object):
    def __init__(self, model, rootdir, workname, *args, **kwargs) -> None:
        self.CHECKPOINT_OPTS = CHECKPOINT_OPTS
        self.TRAINING_OPTIONS = TRAINING_OPTIONS
        self.EXPERIMENT_OPTIONS = EXPERIMENT_OPTIONS
        self.model = model
        self.model_version = ''
        self.rootdir = rootdir
        self._valid_states = ('inference', 'train', 'val', 'test')
        self.workdir_enabled = False
        self.RESERVED_KEYS = ['DATE_OF_CREATION', 'MODEL', 'LR', 'LOSS', 'ACC', 'ID']
        self.tensor_scalar_items = []
        if not os.path.isdir(self.rootdir):
            raise Exception('Rootdir set is not a directory')
        self.db_name = 'experiment_db.sqlite'
        self.db_dir = os.path.join(self.rootdir, self.db_name)
        self.db = sqlite_tools.sq(self.db_dir)
        self.key = {}
        self.init_workname(workname)
        self._state = 'train'

    def __set_property_attr_twin__(self, name, value, **kwargs):
        setattr(self, '_' + name, value)
        obj = meters.get_scalar_warehouse(**kwargs)
        setattr(self, name + '_', obj)

    def __set_property_attr__(self, name, value):
        setattr(self, '_' + name, value)

    def __set_property__(self, name, set_function_name, get_function_name, del_function_name):
        set_function_name_base = set_function_name
        get_function_name_base = get_function_name
        set_function_name += '_' + name
        get_function_name += '_' + name
        setattr(self, set_function_name, partial(getattr(self, set_function_name_base), name=name))
        setattr(self, get_function_name, partial(getattr(self, get_function_name_base), name=name))

        if del_function_name is not None:
            del_function_name_base = del_function_name
            del_function_name += '_' + name
            setattr(self, del_function_name, partial(getattr(self, del_function_name_base), name=name))

        if del_function_name is None:
            setattr(self.__class__, name, property(getattr(self, get_function_name),
                                                   getattr(self, set_function_name)))
        else:
            setattr(self.__class__, name, property(getattr(self, get_function_name),
                                                   getattr(self, set_function_name),
                                                   getattr(self, del_function_name)))

    def set_property(self, name, value, set_function_name, get_function_name, del_function_name):
        if name in self.RESERVED_KEYS:
            raise ValueError('%s is a reserved key' % name)
        self.__set_property_attr__(name, value)
        self.__set_property__(name, set_function_name, get_function_name, del_function_name)

    def set_property_twin(self, name, value, set_function_name, get_function_name, del_function_name):
        if name in self.RESERVED_KEYS:
            raise ValueError('%s is a reserved key' % name)
        self.tensor_scalar_items.append(name)
        self.__set_property_attr_twin__(name, value)
        self.__set_property__(name, set_function_name, get_function_name, del_function_name)

    @staticmethod
    def get_f(self, name):
        return getattr(self, '_' + name)

    @staticmethod
    def set_f(self, val, name):
        if name == 'acc':
            val, table = val

        setattr(self, '_' + name, val)
        pointer = getattr(self, name + '_')
        if torch.is_tensor(val):
            item = val.item()
        else:
            item = val
        if self.iterating:
            pointer[self.state].send(item)
        else:
            if name == 'loss':
                if self.state == 'train':
                    self.key['LOSS'] = item
                elif self.state == 'val':
                    self.key['VLOSS'] = item
            elif name == 'acc':
                if self.state == 'train':
                    self.key['ACC'] = item
                elif self.state == 'val':
                    self.key['VACC'] = item
        if self.tensorboard_enabled:

            if self.iterating:
                if pointer[self.state]['iter'].enabled and self.metrics[name][self.state].on_the_fly:
                    self.writer.add_scalars('%s_iter' % name, {self.state: item}, self.absolute_iter)
            else:
                if pointer[self.state]['epoch'].enabled:
                    self.writer.add_scalars('%s_epoch' % name, {self.state: item}, self.epoch)

    @staticmethod
    def del_f(self, name):
        delattr(self, '_' + name)
        delattr(self, name + '_')

    def set_tensor_scalar_item(self, var_name):
        self.set_property_twin(var_name, None, 'set_f', 'get_f', 'del_f')

    @property
    def state(self):
        return str(self._state)  # Protects the variable

    @state.setter
    def state(self, state):
        assert state in self._valid_states
        if hasattr(self, '_state'):
            self.prevstate = self._state
        else:
            self.prevstate = 'train'
        self._state = state

    @property
    def workname(self):
        return self._workname

    @property
    def training(self):
        return self.state == 'train'

    def init_workname(self, workname):
        if workname is not None:
            self.resume = self.db.exists(workname)
            if self.resume:
                self.workname = workname
            else:
                if not os.path.exists(workname) or not os.path.isfile(workname):
                    raise Exception('Workname: "{}" should point to pre-trained weights or to be None'.format(workname))
                else:
                    self._workname = workname
        else:
            self._workname = workname
            self._workdir = None
            self.resume = False

    @workname.setter
    def workname(self, workname):
        if workname is not None:
            self._workname = workname
            self.workdir = os.path.join(self.rootdir, self._workname)
        else:
            self._workdir = None
            self.resume = False

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, path):
        assert isinstance(path, str)
        print('Setting workdir at {}'.format(path))
        self._workdir = path
        self.workdir_enabled = True

    def __update_db__(self):

        if type(self.key) == dict and self.key is not False:
            self.db.update(self.key)
        else:
            raise Exception('Trying to update a Null key dictionary')

    def __repr__(self):
        return 'PyTorch framework created {0} at {1}'.format(self.key['DATE_OF_CREATION'], self.rootdir)

    def __call__(self):
        self.print_key()

    @assert_workdir
    def __setloggers__(self, **kwargs):
        ptutils.setup_logger('train_iter_log', os.path.join(self.workdir, 'train_iter_log.txt'), **kwargs)
        self.train_iter_logger = logging.getLogger('train_iter_log')

        ptutils.setup_logger('val_epoch_log', os.path.join(self.workdir, 'val_epoch_log.txt'), **kwargs)
        self.val_epoch_logger = logging.getLogger('val_epoch_log')

        ptutils.setup_logger('error_log', os.path.join(self.workdir, 'err.txt'), **kwargs)
        self.err_logger = logging.getLogger('error_log')

    def __setup_experiment__(self, **kwargs):
        self.start_epoch = 0
        self.absolute_iter = 0
        now = datetime.datetime.now()
        if self.EXPERIMENT_OPTIONS.experiment_name == 'datatime':
            if self.EXPERIMENT_OPTIONS.experiment_name_complexity == 0:
                self.workname = str(now)[:19]
            else:
                self.workname = str(now)
        else:
            if self.EXPERIMENT_OPTIONS.experiment_name_complexity == 0:
                self.workname = str(uuid.uuid4())[:7]
            else:
                self.workname = str(uuid.uuid4())
        create_folder(self.workdir)
        if self.EXPERIMENT_OPTIONS.enable_model_logger:
            ptutils.setup_logger('model_logger', os.path.join(self.workdir, 'model_architecture.txt'), writemode='w')
            self.model_logger = logging.getLogger('model_logger')
            self.model_logger.info('Model Version: {0}'.format(self.model_version))
            self.model_logger.info(self.model)
        self.__setloggers__(writemode='w', to_console=False, level=logging.INFO)

        self.key = {'ID': self.workname,
                    'MODEL': self.model_version,
                    'DATE_OF_CREATION': now.strftime("%Y-%m-%d %H:%M"),
                    }
        self.db.insert_value(self.key)  # Creates the table
        self.loaded_model = True

    def print_info(self, log):
        """

        :param log: Logging logger in which to parse info
        :type log: logging.logger
        :return: None
        """

        result = subprocess.Popen(["nvidia-smi", "--format=csv",
                                   "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        nvidia = result.stdout.readlines().copy()
        nvidia = [str(x) for x in nvidia]
        nvidia = [x[2:-3] + '\r\t' for x in nvidia]
        acum = ''
        for x in nvidia:
            acum = acum + x

        log.info('\r\t __Python VERSION: {0} \r\t'
                 '__pyTorch VERSION: {1} \r\t'
                 '__CUDA VERSION: {2}\r\t'
                 '__CUDNN VERSION: {3} \r\t'
                 '__Number CUDA Devices: {4} \r\t'
                 '__Devices: {5}'
                 'Active CUDA Device: GPU {6} \r\t'
                 'Available devices {7} \r\t'
                 'Current cuda device {8} \r\t'.format(sys.version, torch.__version__, torch.version.cuda,
                                                       torch.backends.cudnn.version(), torch.cuda.device_count(),
                                                       acum, torch.cuda.current_device(), torch.cuda.device_count(),
                                                       torch.cuda.current_device()))


class pytorchfw(framework):
    def __init__(self, model: torch.nn.Module, rootdir: str, workname: (str, None), main_device: (int, str),
                 trackgrad: bool, debug=False) -> None:
        """
        PyTorch Framework init function.

        :param model: torch model
        :type model: torch.nn.Module
        :param rootdir: Path to root directory for all experiments
        :type rootdir: str
        :param workname: Experiment name to resume from or path to pretrained weights.
        :type workname: str, None
        :param main_device: GPU taken as main gpu. 'cpu' enables cpu mode.
        :type main_device: str,int,torch.device

        """
        self.debug = debug
        if self.debug:
            super(pytorchfw, self).__init__(Debugger(model, raise_exception=True), rootdir, workname)
        else:
            super(pytorchfw, self).__init__(model, rootdir, workname)

        self.checkpoint_name = 'checkpoint.pth'
        self.cuda = torch.cuda.is_available()
        self._main_device = 'cuda:{0}'.format(int(main_device)) if self.cuda and main_device != 'cpu' else 'cpu'
        self.main_device = self._main_device

        if main_device != 'cpu':
            self.model.to(self.main_device)
        self.inizilizable_layers = [self.model]
        self.tensorboard_enabled = True
        self.metrics = BaseDict()
        self.prevstate = 'train'
        self.iterating = False
        self.scheduler = None
        self.acc_ = classitems.NullItem()
        self.loaded_model = False
        self.trackgrad = trackgrad
        self.assertion_variables = ['initializer', 'EPOCHS', 'optimizer', 'criterion', 'LR', 'dataparallel']
        if bool(trackgrad):
            self.tracker = tracegrad(trackgrad, self.model, verbose=False)

    def init_loss(self):

        self.metrics['loss'] = meters.get_loss_meter(self.criterion)
        self.set_tensor_scalar_item('loss')

    def init_acc(self, labels, on_the_fly):
        self.metrics['acc'] = meters.get_acc_meter(labels, self.criterion, on_the_fly)
        self.set_tensor_scalar_item('acc')

    def __enter__(self):
        self.prevmodelstate = self.model.training

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.iterating:
            self.state = self.prevstate
        self.iterating = False
        return False

    @property
    def absolute_iter(self):
        return self._absolute_iter

    @absolute_iter.setter
    def absolute_iter(self, value):
        self._absolute_iter = value
        self.key['ITERATIONS'] = self._absolute_iter

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, value):
        self._acc = value
        if value is not None:
            self.acc_.update_epoch(self.state)
            if self.state == 'train':
                self.key['ACC'] = value
            elif self.state == 'val':
                self.key['VACC'] = value
            if self.tensorboard_enabled:
                if self.acc_.tuple[self.state].epoch_array.enabled:
                    self.writer.add_scalars('accuracy',
                                            {self.state: value},
                                            self.epoch)

                if self.acc_.tuple[self.state].cmflag:
                    self.writer.add_figure('confusion matrix',
                                           self.acc_.plot_confusion_matrix(self.state)[0],
                                           self.epoch)

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        self._epoch = value
        self.key['EPOCH'] = self._epoch

    @property
    def main_device(self):
        return self._main_device

    @main_device.setter
    def main_device(self, value):
        self._main_device = torch.device(value)

    @assert_workdir
    def _set_writer(self, **kwargs):
        if not kwargs:
            if TENSORBOARDX:
                kwargs = {'logdir': os.path.join(self.workdir, 'tensorboard')}
                self.summary_writer_path = kwargs['logdir']
            else:
                kwargs = {'log_dir': os.path.join(self.workdir, 'tensorboard')}
                self.summary_writer_path = kwargs['log_dir']

        self.writer = SummaryWriter(**kwargs)

    def _allocate_tensor(self, x, device=None):
        if device is None:
            device = self.main_device
        else:
            device = torch.device(device)
        if self.dataparallel:
            return x
        else:
            if isinstance(x, list):
                return [i.to(device) for i in x]
            elif isinstance(x, tuple):
                return tuple([i.to(device) for i in x])
            else:
                return x.to(device)

    def set_model_training(self, flag) -> bool:
        """
        Trigger function to enable or disable model.train() and model.eval()

        :param flag: True --> activate model.train() mode. False --> activate model.eval() mode
        :type flag: bool
        :returns: bool
        """
        if flag:
            self.model.train()
        else:
            self.model.eval()
        return flag

    def _allocate_to_outputdevice(self, x, default):
        if not hasattr(self, 'outputdevice'):
            return self._allocate_tensor(x, self.outputdevice)
        else:
            return self._allocate_tensor(x, default)

    def _loadcheckpoint(self):
        try:
            directory = os.path.join(self.workdir, 'best' + self.checkpoint_name)
            assert os.path.isfile(directory)
        except AssertionError:
            directory = os.path.join(self.workdir, self.checkpoint_name)
            assert os.path.isfile(directory)
        except AssertionError:
            raise FileNotFoundError("=> No checkpoint found at '{}'".format(directory))

        self.__setloggers__(writemode='a', to_console=False, level=logging.INFO)

        print("=> Loading checkpoint '{}'".format(directory))
        checkpoint = torch.load(directory, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch']
        self.load_model(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Detecting scalars...')
        for key in checkpoint['tsi']:
            setattr(self, key + '_', checkpoint['tsi'][key])
            print('>' + key + ' loaded!')
        self.key.update(checkpoint['key'])
        self.absolute_iter = checkpoint['iter']
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(directory, checkpoint['epoch']))

    @checkpoint_on_key
    @assert_workdir
    def save_checkpoint(self, metric, criteria, filename=None):
        """
        Save checkpoint function. By default saves last epoch for training and best validation epoch. If there is
        no validation then best train epoch.

        :param filename: Custom filename to save weights. Defautl is framework.checkpoint_name
        :type filename: str
        :return: None
        """
        state = {
            'epoch': self.epoch + 1,
            'iter': self.absolute_iter + 1,
            'arch': self.model_version,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'key': self.key,
            'scheduler': self.scheduler.state_dict()
        }
        tsi = {}
        for key in self.tensor_scalar_items:
            tsi.update({key: getattr(self, key + '_')})
        state.update({'tsi': tsi})
        if filename is None:
            filename = os.path.join(self.workdir, self.checkpoint_name)

        elif isinstance(filename, str):
            filename = os.path.join(self.workdir, filename)
        print('Saving checkpoint at : {}'.format(filename))
        torch.save(state, filename)
        if getattr(self, metric + '_')[self.state].is_best(criteria):
            shutil.copyfile(filename, os.path.join(self.workdir, 'best' + self.checkpoint_name))
        print('Checkpoint saved successfully')

    def checkpoint(self, metric='loss', criteria=[min, lambda epoch, optimal: optimal > epoch], filename=None):

        return partial(self.save_checkpoint, metric=metric, criteria=criteria, filename=filename)

    def load_model(self, directory, strict_loading=True, from_checkpoint=False, **kwargs):
        """
        Load model function which allows to load from a python dict, path to weights or checkpoint.
        If called enables framework.loaded_model = True

        :param directory: Loaded state dict or path to weights or checkpoint.
        :type directory: dict,str
        :param strict_loading: PyTorch strict_loading flag. Impose exact matching between loaded weights and model.
        :type strict_loading: bool
        :param from_checkpoint: Forces function to interpret given weights as checkpoint dictionary.
        :type   from_checkpoint: bool
        :return: None
        """
        print('Loading pre-trained weights')
        if isinstance(directory, dict):
            state_dict = directory
        else:
            state_dict = torch.load(directory, map_location=lambda storage, loc: storage)
        if 'checkpoint' in directory or from_checkpoint:
            state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict, strict=strict_loading)
        self.loaded_model = True

    def error_handled_forward(self, c, inputs):
        try:
            output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
        except InfError as ex:
            raise type(ex)(
                str(
                    ex) + 'Inf detected! Iteration %d, epoch %d. Mode %s' % (c, self.epoch, self.state)).with_traceback(
                sys.exc_info()[2])
        except NaNError as ex:
            raise type(ex)(
                str(
                    ex) + 'Nan detected! Iteration %d, epoch %d. Mode %s' % (c, self.epoch, self.state)).with_traceback(
                sys.exc_info()[2])
        except Exception as ex:
            raise type(ex)(
                str(
                    ex) + 'Error raised. Iteration %d, epoch %d. Mode %s' % (c, self.epoch, self.state)).with_traceback(
                sys.exc_info()[2])
        return output

    def run_epoch(self, dataloader, metrics=[], checkpoint=lambda: None, allocate_input=True,
                  allocate_gt=True, send=('pred', 'gt')):
        j = -1
        iterations = len(dataloader)
        if torch.is_grad_enabled() and 'loss' not in metrics:
            metrics.append('loss')
        with tqdm(dataloader,
                  desc='|| {2} || Epoch: [{0}/{1}]'.format(self.epoch, self.EPOCHS, self.state)) as pbar, ctx_iter(
            self):
            for gt, inputs, vs in pbar:
                try:
                    j += 1
                    self.absolute_iter += 1
                    if allocate_input:
                        inputs = self._allocate_tensor(inputs)

                    pred = self.error_handled_forward(self.absolute_iter, inputs)

                    if allocate_gt:
                        if hasattr(self, 'outputdevice'):
                            device = torch.device(self.outputdevice)
                        else:
                            if isinstance(pred, (list, tuple)):
                                device = pred[0].device
                            else:
                                device = pred.device
                        gt = self._allocate_tensor(gt, device=device)
                    for metric in metrics:
                        if metric == 'loss' or metric == 'acc':
                            send_ = ('pred', 'gt')
                        else:
                            send_ = send
                        for var in send_:
                            self.metrics[metric][self.state].send(key=var, value=locals()[var])
                        if self.metrics[metric][self.state].on_the_fly:
                            if self.metrics[metric][self.state].redirect is not None:
                                results = self.metrics[metric][self.state].process(*send_, store=False)
                                for key in self.metrics[metric][self.state].redirect:
                                    name = self.metrics[metric][self.state].redirect[key]
                                    setattr(self, name, results[key])
                            else:
                                self.metrics[metric][self.state].process(send_, store=True)

                    # compute gradient and do SGD step
                    if torch.is_grad_enabled():
                        self.optimizer.zero_grad()
                        self.loss.backward()
                        self.gradients()
                        if self.trackgrad and self.tensorboard_enabled:
                            self.writer.add_figure('Gradients',
                                                   self.tracker(self.model.named_parameters()),
                                                   self.absolute_iter)
                        self.optimizer.step()
                    # CHECKPOINT

                    self.tensorboard_writer(self.loss, pred, gt, self.absolute_iter, vs)
                    postfix = {}
                    for metric in metrics:
                        if self.metrics[metric][self.state].redirect is not None:
                            for key in self.metrics[metric][self.state].redirect:
                                name = self.metrics[metric][self.state].redirect[key]
                                scalar = getattr(self, name)
                                if torch.is_tensor(scalar):
                                    scalar = scalar.item()
                                postfix.update({name: scalar})
                    pbar.set_postfix(postfix)
                    # pbar.set_postfix(loss=self.loss.item())

                except Exception as e:
                    try:
                        if self.TRAINING_OPTIONS.enable_backup:
                            self.save_checkpoint(filename=os.path.join(self.workdir, 'checkpoint_backup.pth'))
                    except:
                        if self.TRAINING_OPTIONS.enable_error_logger:
                            self.err_logger.error('Failed to deal with exception. Couldnt save backup at {0} \n'
                                                  .format(os.path.join(self.workdir, 'checkpoint_backup.pth')))
                            self.err_logger.error(str(e))
                    raise e

        for metric in metrics:
            if metric == 'loss' or metric == 'acc':
                send_ = ('gt', 'pred')
            else:
                send_ = send
            if not self.metrics[metric][self.state].on_the_fly:
                if self.metrics[metric][self.state].redirect is not None:
                    results = self.metrics[metric][self.state].process(*send_, store=False)
                    for key in self.metrics[metric][self.state].redirect:
                        name = self.metrics[metric][self.state].redirect[key]
                        setattr(self, name, results[key])
                        getattr(self, name + '_')[self.state].reset()
                else:
                    results = self.metrics[metric][self.state].process(send_, store=True)
            else:
                if self.metrics[metric][self.state].redirect is not None:
                    for key in self.metrics[metric][self.state].redirect:
                        name = self.metrics[metric][self.state].redirect[key]
                        setattr(self, name, getattr(self, name + '_')[self.state].process(store=True))

        # Metrics
        self.__update_db__()
        checkpoint()

    def __atribute_assertion__(self):
        assert hasattr(self, 'assertion_variables')
        try:
            for variable in self.assertion_variables:
                assert hasattr(self, variable)
        except Exception as e:
            # error_logger hasn't been defined at this point, that's why print
            print('Variable assertion failed. Framework requires >{0}< to be defined'.format(variable))
            raise e

    def __initialize_layers__(self):
        if self.inizilizable_layers is None:
            print('Network not automatically initilized')
        else:
            print('Network automatically initilized at : \n ')
            for m in self.inizilizable_layers:
                print('\t' + m.__class__.__name__)
            map(self.init_function, self.inizilizable_layers)

    def _train(self):  # TODO Rewrite _train as generical settings
        # Parse work to _loadcheckpoint in order to
        if self.resume:  # Resume a training stage
            self._loadcheckpoint()
        else:
            if self.workname is None:  # Training from scratch
                self.__setup_experiment__()
                self.__initialize_layers__()

            else:  # Training from pre-trained weights
                path_to_weights = self.workname
                self.__setup_experiment__()

                self.load_model(path_to_weights)

        if self.dataparallel:
            self.model = torch.nn.DataParallel(self.model)
        if TENSORBOARDX:
            self._set_writer(logdir=os.path.join(self.workdir, 'tensorboard'))
        else:
            self._set_writer(log_dir=os.path.join(self.workdir, 'tensorboard'))

    def save_gradients(self, absolute_iter):
        grads = self.tracker.grad(numpy=True)
        grad_path = os.path.join(self.workdir, 'gradient_tracking')
        if not os.path.exists(grad_path):
            os.makedirs(grad_path)
        grad_path = os.path.join(grad_path, 'grad_{0}_{1:06d}.npy'.format(self.workname, absolute_iter))
        np.save(grad_path, grads)

    def set_optim(self, *args, **kwargs):
        if self.args.optimizer == 'Adam':
            return torch.optim.Adam(*args, **kwargs)
        elif self.args.optimizer == 'SGD':
            return torch.optim.SGD(*args, **kwargs)
        elif self.args.optimizer == 'Adadelta':
            return torch.optim.Adadelta(*args, *kwargs)
        elif self.args.optimizer == 'Adagrad':
            return torch.optim.Adadelta(*args, *kwargs)
        elif self.args.optimizer == 'Adamax':
            return torch.optim.Adadelta(*args, *kwargs)
        elif self.args.optimizer == 'ASGD':
            return torch.optim.Adadelta(*args, *kwargs)
        elif self.args.optimizer == 'RMSprop':
            return torch.optim.Adadelta(*args, *kwargs)
        elif self.args.optimizer == 'Rprop':
            return torch.optim.Adadelta(*args, *kwargs)
        else:
            raise Exception('Non considered optimizer. Implement it')

    @set_training
    @config
    def train(self, args):
        """
        Main training function to be overwritten.
        Example:
            database = call_your_db()
            self.train_loader = self.val_loader = torch.utils.data.DataLoader(database, batch_size=self.batch_size)
            for self.epoch in range(self.start_epoch, self.EPOCHS):
                with train(self):
                    self.run_epoch(self.train_iter_logger)
                with val(self):
                    self.run_epoch()
        :param args: Used defined optional arguments to setup training stage.
        :return: Used defined
        """
        NotImplementedError

    def tensorboard_writer(self, loss, output, gt, absolute_iter, visualization):
        """
        Function called after each backpropagation. Allows to pass custom info to tensorboard_writer or to
        compute custom operations over the output such as visualizations or metrics.
        :param loss: Iteration loss.
        :type loss: torch.tensor
        :param output: Network's output (whatever it be)
        :param gt: Ground-truth.
        :param absolute_iter: Absolute iteration
        :type absolute_iter: int
        :param visualization: Visualization / extra information passed
        """

    def infer(self, *inputs):
        NotImplementedError

    def hyperparameters(self):
        """
        Function in which to define ``training`` hyperparameters.
        Example:
            self.initializer = 'xavier'
            self.EPOCHS = 15
            self.LR = 0.001
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR)
            self.criterion = torch.nn.BCELoss().to(self.main_device)
        :return: None
        """
        NotImplementedError

    def gradients(self):
        """
        Function called after backpropagation but before gradient update. Allows to in-place modify gradients before
        updating.
        """

    def set_config(self):
        """
        Function in which to define other necessary parameters such as metrics, wheter to use dataparallel or not,
        batch size etcetera... Consider this function should contain necessary parameters to setup a validation step.
        Example:
            self.batch_size = 2
            self.dataparallel = False
            self.acc_ = classitems.TensorAccuracyItem(['perro', 'gato', 'loro'])
        :return: None
        """
        NotImplementedError


def _test():
    import torch.utils.data

    class toy_example(torch.nn.Module):
        def __init__(self):
            super(toy_example, self).__init__()
            self.module1 = torch.nn.Conv2d(1, 10, 3)
            self.module2 = torch.nn.Conv2d(10, 10, 3)

        def forward(self, x):
            x = self.module1(x)
            x = self.module2(x)
            return torch.sigmoid(x)

    class db(torch.utils.data.Dataset):
        def __len__(self):
            return 30

        def __getitem__(self, idx):
            return torch.randint(0, 2, (10, 6, 6)).float(), [torch.randint(0, 5, (1, 10, 10)).float()], []

    class toy_fw(pytorchfw):
        def hyperparameters(self):
            self.hihihi = 5
            self.initializer = 'xavier'
            self.EPOCHS = 15
            self.LR = 0.000000000001
            # Def optimizer self.optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR)
            # Def criterion self.criterion
            self.criterion = torch.nn.BCELoss().to(self.main_device)

        def set_config(self):
            self.batch_size = 2
            self.dataparallel = False
            self.acc_ = classitems.TensorAccuracyItem(['perro', 'gato', 'casa', 'miau', 'guau'])

        @config
        @set_training
        def train(self):
            datab = db()
            self.train_loader = self.val_loader = torch.utils.data.DataLoader(datab, batch_size=self.batch_size)
            for self.epoch in range(self.start_epoch, self.EPOCHS):
                with train(self):
                    self.run_epoch(self.train_iter_logger)
                with val(self):
                    self.run_epoch()

    fw = toy_fw(toy_example(), './', None, 0, False)
    return fw
