#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Juan Montesinos"
__version__ = "0.2.3"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import torch
import os
import uuid
import numpy as np
import logging
import sys
import time
import shutil
import datetime
from collections import OrderedDict
import subprocess
from . import utils as ptutils, network_initialization, sqlite_tools
from .traceback_gradients import tracegrad
from . import classitems
from .classitems import GradientPlotter as tracegrad
from . import *
from tqdm import tqdm
from functools import wraps

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from functools import partial

LOGGING_FORMAT_B = "[%(filename)s: %(funcName)s] %(message)s]"
LOGGIN_FORMAT_A = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"


def set_training(func):
    @wraps(func)
    def inner(*args, **kwargs):
        self = args[0]
        self.hyperparameters()
        self.__atribute_assertion__()
        if not hasattr(self, 'init_function'):
            self.init_function = partial(network_initialization.init_weights, init_type=self.initializer)
        self.scheduler = classitems.Scheduler(self.scheduler)
        self._train()

        self.key['LR'] = self.LR  # TODO Learning rate reduction by hand does not overwrite optimizer
        self.key['OPTIMIZER'] = str(self.optimizer)
        self.key['MODEL'] = self.model_version
        self.key['SCHEDULER'] = str(self.scheduler)
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
    def __init__(self, model, rootdir, workname, *args, **kwargs):
        self.model = model
        self.model_version = ''
        self.rootdir = rootdir
        self._valid_states = ('inference', 'train', 'val', 'test')
        self.workdir_enabled = False
        self.RESERVED_KEYS = ['DATE_OF_CREATION', 'MODEL', 'LR', 'LOSS', 'ACC', 'ID']
        if not os.path.isdir(self.rootdir):
            raise Exception('Rootdir set is not a directory')
        self.db_name = 'experiment_db.sqlite'
        self.db_dir = os.path.join(self.rootdir, self.db_name)
        self.db = sqlite_tools.sq(self.db_dir)
        self.key = {}
        self.init_workname(workname)
        self._state = 'train'

    def __set_property_attr_twin__(self, name, value):
        setattr(self, '_' + name, value)
        setattr(self, name + '_', value)

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
            setattr(self, get_function_name, partial(getattr(self, del_function_name_base), name=name))

        if del_function_name is None:
            setattr(self.__class__, name, property(getattr(self, get_function_name),
                                                   getattr(self, set_function_name)))
        else:
            setattr(self.__class__, name, property(getattr(self, get_function_name),
                                                   getattr(self, set_function_name),
                                                   getattr(self, del_function_name)))

    def set_property(self, name, value, set_function_name, get_function_name, del_function_name):
        self.__set_property_attr__(name, value)
        self.__set_property__(name, set_function_name, get_function_name, del_function_name)

    def set_property_twin(self, name, value, set_function_name, get_function_name, del_function_name):
        self.__set_property_attr_twin__(name, value)
        self.__set_property__(name, set_function_name, get_function_name, del_function_name)

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
        self.workname = str(uuid.uuid4())[:7]
        self.workdir = os.path.join(self.rootdir, self.workname)
        create_folder(self.workdir)
        ptutils.setup_logger('model_logger', os.path.join(self.workdir, 'model_architecture.txt'), writemode='w')
        self.model_logger = logging.getLogger('model_logger')
        self.model_logger.info('Model Version: {0}'.format(self.model_version))
        self.model_logger.info(self.model)
        self.__setloggers__(writemode='w', to_console=False)

        self.key = {'ID': self.workname, 'MODEL': self.model_version,
                    'DATE_OF_CREATION': now.strftime("%Y-%m-%d %H:%M")}
        self.db.insert_value(self.key)  # Creates the table
        self.loaded_model = True

    def print_info(self, log):

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
    def __init__(self, model, rootdir, workname, main_device, trackgrad):
        super(pytorchfw, self).__init__(model, rootdir, workname)
        self.checkpoint_name = 'checkpoint.pth'
        self.cuda = torch.cuda.is_available()
        self._main_device = 'cuda:{0}'.format(int(main_device)) if self.cuda and main_device != 'cpu' else 'cpu'
        self.main_device = self._main_device
        if main_device != 'cpu':
            self.model.to(self.main_device)
        self.inizilizable_layers = [self.model]
        self.loss_ = classitems.TensorScalarItem()
        self.tensorboard_enabled = True
        self.prevstate = 'train'
        self.iterating = False
        self.scheduler = None
        self.acc_ = classitems.NullItem()
        self.loaded_model = False
        self.trackgrad = trackgrad
        self.assertion_variables = ['initializer', 'EPOCHS', 'optimizer', 'criterion', 'LR', 'dataparallel']
        if bool(trackgrad):
            self.tracker = tracegrad(trackgrad, self.model, verbose=False)

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        self.prevmodelstate = self.model.training

        if self.state == 'train':
            torch._C.set_grad_enabled(True)
            self.model.train()

        else:
            torch._C.set_grad_enabled(False)
            self.model.eval()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(self.prev)
        if not self.iterating:
            self.state = self.prevstate
        self.iterating = False
        self.set_model_training(self.prevmodelstate)
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
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        self._loss = loss
        if self.iterating:
            self.loss_(loss, self.state)
        else:
            if self.state == 'train':
                self.key['LOSS'] = loss
            elif self.state == 'val':
                self.key['VLOSS'] = loss

        if self.tensorboard_enabled:
            loss = loss.item()
            if self.iterating:
                if self.loss_.data.tuple[self.state].array.enabled:
                    self.writer.add_scalars('loss', {self.state: loss}, self.absolute_iter)
            else:
                if self.loss_.data.tuple[self.state].epoch_array.enabled:
                    self.writer.add_scalars('loss_epoch', {self.state: loss}, self.epoch)

    @property
    def main_device(self):
        return self._main_device

    @main_device.setter
    def main_device(self, value):
        self._main_device = torch.device(value)

    @assert_workdir
    def _set_writer(self, **kwargs):
        if not kwargs:
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

    def set_model_training(self, flag):
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

        self.__setloggers__(writemode='a', to_console=False)

        print("=> Loading checkpoint '{}'".format(directory))
        checkpoint = torch.load(directory, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch']
        self.load_model(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.loss_.constructor(checkpoint['loss'])  # TODO verify this is ok for savecheckpoint
        self.acc_.constructor()
        self.absolute_iter = checkpoint['iter']
        self.key = checkpoint['key']
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(directory, checkpoint['epoch']))

    @checkpoint_on_key
    @assert_workdir
    def save_checkpoint(self, filename=None):
        state = {
            'epoch': self.epoch + 1,
            'iter': self.absolute_iter + 1,
            'arch': self.model_version,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_,
            'key': self.key,
            'scheduler': self.scheduler.state_dict()
        }
        if filename is None:
            filename = os.path.join(self.workdir, self.checkpoint_name)

        elif isinstance(filename, str):
            filename = os.path.join(self.workdir, filename)
        print('Saving checkpoint at : {}'.format(filename))
        torch.save(state, filename)
        if self.loss_.data.is_best():
            shutil.copyfile(filename, os.path.join(self.workdir, 'best' + self.checkpoint_name))
        print('Checkpoint saved successfully')

    def load_model(self, directory, strict_loading=True, from_checkpoint=False, **kwargs):
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
        # TODO initilize this variable

    def run_epoch(self, *args, **kwargs):
        if self.state == 'train':
            self.train_epoch(*args, **kwargs)
        elif self.state == 'val':
            self.validate_epoch(*args, **kwargs)
        elif self.state == 'inference':
            self.inference(*args, **kwargs)
        elif self.state == 'test':
            self.test(*args, **kwargs)
        else:
            raise ValueError('Not existing forwarding mode')

    def train_epoch(self, logger):
        j = 0
        self.train_iterations = len(iter(self.train_loader))
        with tqdm(self.train_loader, desc='Epoch: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar, ctx_iter(self):
            for gt, inputs, visualization in pbar:
                try:
                    self.absolute_iter += 1

                    inputs = self._allocate_tensor(inputs)

                    output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)

                    if hasattr(self, 'outputdevice'):
                        device = torch.device(self.outputdevice)  # TODO keep an eye here
                    else:
                        if isinstance(output, (list, tuple)):
                            device = output[0].device
                        else:
                            device = output.device

                    gt = self._allocate_tensor(gt, device=device)
                    self.acc_('train', gt, output)
                    self.loss = self.criterion(output, gt)
                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    self.loss.backward()

                    self.gradients()
                    if self.trackgrad:
                        self.writer.add_figure('Gradients',
                                               self.tracker(self.model.named_parameters()),
                                               self.absolute_iter)
                    self.optimizer.step()
                    pbar.set_postfix(loss=self.loss.item())

                    self.loss_.data.print_logger(self.epoch, j, self.train_iterations, logger)
                    j += 1

                except Exception as e:
                    try:
                        self.save_checkpoint(filename=os.path.join(self.workdir, 'checkpoint_backup.pth'))
                    except:
                        self.err_logger.error('Failed to deal with exception. Couldnt save backup at {0} \n'
                                              .format(os.path.join(self.workdir, 'checkpoint_backup.pth')))
                    self.err_logger.error(str(e))
                    raise e
        self.loss = self.loss_.data.update_epoch(self.state)
        self.acc = self.acc_.get_acc('train')
        self.__update_db__()
        self.save_checkpoint()
        # probably does not require dense tracking

    def validate_epoch(self):

        with tqdm(self.val_loader, desc='Validation: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar, ctx_iter(
                self):
            for gt, inputs, visualization in pbar:

                inputs = self._allocate_tensor(inputs)

                output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                self.acc_('val', gt, output)
                if hasattr(self, 'outputdevice'):
                    device = torch.device(self.outputdevice)  # TODO keep an eye here
                else:
                    if isinstance(output, (list, tuple)):
                        device = output[0].device
                    else:
                        device = output.device

                gt = self._allocate_tensor(gt, device=device)
                self.loss = self.criterion(output, gt)
        self.loss = self.loss_.data.update_epoch(self.state)
        self.acc = self.acc_.get_acc('val')

    def __atribute_assertion__(self):
        assert hasattr(self, 'assertion_variables')
        try:
            for variable in self.assertion_variables:
                assert hasattr(self, variable)
        except Exception as e:
            self.err_logger.error(str(e))
            self.err_logger.error('Variable assertion failed. Framework requires >{0}< to be defined'.format(variable))
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
        self._set_writer(log_dir=os.path.join(self.workdir, 'tensorboard'))

    def save_gradients(self, absolute_iter):
        grads = self.tracker.grad(numpy=True)
        grad_path = os.path.join(self.workdir, 'gradient_tracking')
        if not os.path.exists(grad_path):
            os.makedirs(grad_path)
        grad_path = os.path.join(grad_path, 'grad_{0}_{1:06d}.npy'.format(self.workname, absolute_iter))
        np.save(grad_path, grads)

    @set_training
    @config
    def train(self, args):
        NotImplementedError

    def tensorboard_writer(self, loss, output, gt, absolute_iter, visualization):
        pass

    def infer(self, *inputs):
        NotImplementedError

    def hyperparameters(self):
        NotImplementedError

    def gradients(self):
        pass

    def set_config(self):
        NotImplementedError


def test():
    import torch.utils.data

    class toy_example(torch.nn.Module):
        def __init__(self):
            super(toy_example, self).__init__()
            self.module1 = torch.nn.Conv2d(1, 10, 3)
            self.module2 = torch.nn.Conv2d(10, 10, 3)

        def forward(self, x):
            x = self.module1(x)
            x = self.module2(x)
            return torch.nn.functional.sigmoid(x)

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
