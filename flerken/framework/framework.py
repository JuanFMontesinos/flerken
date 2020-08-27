import torch
from tqdm import tqdm

from .experiment import FileManager, Arxiv, experiment_cfg
from .allocator import Allocator
from .model import Model
from .debug import NaNError, InfError, Debugger
from . import classitems, meters

import os
import shutil
import sys
import datetime
from functools import partial
from typing import Union


class DinamicProperty(object):
    def __init__(self):
        self.RESERVED_KEYS = ['DATE_OF_CREATION', 'MODEL', 'LR', 'LOSS', 'ACC', 'ID']
        self.tensor_scalar_items = []

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


class Trainer(DinamicProperty):
    def __init__(self, main_device, model: torch.nn.Module, dataparallel: bool = False, input_shape=None):
        super().__init__()
        self._enter_called = False
        self.iterating = False
        self.tensorboard_enabled = True
        self.metrics = experiment_cfg()['metrics_dicttype']()
        self.acc_ = classitems.NullItem()
        self.start_epoch = 0
        self.absolute_iter = 0
        self.alloc = Allocator(main_device, dataparallel)
        self.model = Model(model, input_shape)
        self.dataparallel = dataparallel
        self.shape = input_shape
        self.IO: FileManager

    # SHORTCUTS as @property
    @property
    def model(self) -> torch.nn.Module:
        return self._model.model

    @model.setter
    def model(self, value: Model):
        self._model = value

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, val):
        self._scheduler = classitems.Scheduler(val)

    @property
    def cfg(self):
        return self.IO.cfg_files

    def model_ex(self):
        return self._model

    def init_loss(self):
        self.metrics['loss'] = meters.get_loss_meter(self.criterion)
        self.set_tensor_scalar_item('loss')

    def init_acc(self, labels, on_the_fly):
        self.metrics['acc'] = meters.get_acc_meter(labels, self.criterion, on_the_fly)
        self.set_tensor_scalar_item('acc')

    def set_tensor_scalar_item(self, var_name):
        self.set_property_twin(var_name, None, 'set_f', 'get_f', 'del_f')

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

        if self.tensorboard_enabled:

            if self.iterating:
                if pointer[self.state]['iter'].enabled and self.metrics[name][self.state].on_the_fly:
                    self.IO.writer.add_scalars('%s_iter' % name, {self.state: item}, self.absolute_iter)
            else:
                if pointer[self.state]['epoch'].enabled:
                    self.IO.writer.add_scalars('%s_epoch' % name, {self.state: item}, self.epoch)

    @staticmethod
    def del_f(self, name):
        delattr(self, '_' + name)
        delattr(self, name + '_')

    def _atribute_assertion(self):
        assert hasattr(self, 'assertion_variables')
        try:
            for variable in self.assertion_variables:
                assert hasattr(self, variable)
        except Exception as e:
            # error_logger hasn't been defined at this point, that's why print
            print('Variable assertion failed. Framework requires >{0}< to be defined'.format(variable))
            raise e

    def init_metrics(self):
        self.init_loss()

    def __enter__(self):

        self.assertion_variables = ['EPOCHS', 'optimizer', 'criterion', 'dataparallel']
        self._atribute_assertion()
        self.init_metrics()
        self.IO.write_cfg()  # Writes all cfg files and create set metadata_dir
        self.IO._set_writer()  # accepts kwargs, defines tb writer
        if self._model.with_summary():
            path = os.path.join(self.IO.metadata_dir, 'model.txt')
            with open(path, 'w') as file:
                try:
                    file.write(self._model.get_summary(self.alloc.main_device))
                except Exception as ex:
                    file.write(str(self.model))
        path = os.path.join(self.IO.metadata_dir, 'system_info.txt')
        with open(path, 'w') as file:
            file.write(self.IO.get_sys_info())
        if self.IO.resume_from is not None:
            self.load_checkpoint(self.IO.resume_from)
        else:
            self._model._initialize_layers()
        self.IO._enter_called = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, InfError):
            exc_val.args = (*exc_val.args, 'Inf detected! Epoch %d. Mode %s' % (self.epoch, str(self.model.training)))
        elif isinstance(exc_val, NaNError):
            exc_val.args = (*exc_val.args, 'NaN detected! Epoch %d. Mode %s' % (self.epoch, str(self.model.training)))
        elif isinstance(exc_val, KeyboardInterrupt):
            self._save_checkpoint('KeyboardInterrupt_checkpoint.pth')
        elif not hasattr(self, 'epoch'):
            pass
        elif isinstance(exc_val, Exception):
            exc_val.args = (*exc_val.args, 'Error detected! Epoch %d. Mode %s' % (self.epoch, str(self.model.training)))
        self._enter_called = False
        self.IO.writer.close()  # Closing tensorboard writer

    def loader_mapping(self, x):
        (gt, inputs, vs) = x
        if isinstance(inputs, torch.Tensor):

            return {'gt': gt, 'inputs': [inputs], 'vs': vs}
        else:
            return {'gt': gt, 'inputs': inputs, 'vs': vs}

    def pred_mapping(self, x):
        return x

    def backpropagate(self, debug):
        self.optimizer.zero_grad()
        if debug:
            assert not torch.isnan(self.loss).any()
            assert not torch.isinf(self.loss).any()
        self.loss.backward()
        self.gradients()
        self.optimizer.step()

    def gradients(self):
        pass

    def hook(self, vrs):
        pass

    def run_epoch(self, dataloader, state, backprop, metrics=[], checkpoint=lambda: None, allocate_input=True,
                  allocate_gt=True, send={}, debug=False):
        self.state = state
        j = -1
        iterations = len(dataloader)
        if torch.is_grad_enabled() and 'loss' not in metrics:
            metrics.append('loss')
        self.set_model_training(backprop)
        if backprop:
            grad_ctx_manager = torch.enable_grad
        else:
            grad_ctx_manager = torch.no_grad

        self.iterating = True
        with tqdm(dataloader,
                  desc='|| {2} || Epoch: [{0}/{1}]'.format(self.epoch, self.EPOCHS, self.state)) as pbar:
            # for gt, inputs, vs in pbar:
            vrs = {}
            for x in pbar:
                with torch.no_grad():
                    vrs.update(self.loader_mapping(x))
                    j += 1
                    self.absolute_iter += 1
                    if allocate_input:
                        vrs['inputs'] = self.alloc(vrs['inputs'])
                with grad_ctx_manager():
                    if debug:
                        assert torch.is_grad_enabled() == backprop
                    vrs['pred'] = self.model(*vrs['inputs'])
                    vrs.update(self.pred_mapping(vrs))
                    if allocate_gt:
                        if isinstance(vrs['pred'], (list, tuple)):
                            device = vrs['pred'][0].device
                        else:
                            device = vrs['pred'].device
                        vrs['gt'] = self.alloc(vrs['gt'], device=device)
                    for metric in metrics:
                        send_ = send.get(metric, ('pred', 'gt'))
                        for var in send_:
                            self.metrics[metric][self.state].send(key=var, value=vrs[var])
                        if self.metrics[metric][self.state].on_the_fly:
                            if self.metrics[metric][self.state].redirect is not None:
                                results = self.metrics[metric][self.state].process(*send_, store=False)
                                for key in self.metrics[metric][self.state].redirect:
                                    name = self.metrics[metric][self.state].redirect[key]
                                    setattr(self, name, results[key])
                            else:
                                self.metrics[metric][self.state].process(send_, store=True)

                    # compute gradient and do SGD step
                    if backprop:
                        self.backpropagate(debug)

                self.hook(vrs)
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
        self.iterating = False
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
        checkpoint()

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

    def _save_checkpoint(self, filename: str):
        """
        Save checkpoint function. By default saves last epoch for training and best validation epoch. If there is
        no validation then best train epoch.

        :param filename: Custom filename to save weights.
        :type filename: str
        :return: None
        """
        state = {
            'epoch': self.epoch + 1,
            'iter': self.absolute_iter + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        tsi = {}
        for key in self.tensor_scalar_items:
            tsi.update({key: getattr(self, key + '_')})
        state.update({'tsi': tsi})
        if filename is None:
            filename = os.path.join(self.IO.metadata_dir, '%04d' % self.epoch + '_checkpoint.pth')

        elif isinstance(filename, str):
            filename = os.path.join(self.IO.metadata_dir, filename)
        print('Saving checkpoint at : {}'.format(filename))
        torch.save(state, filename)
        return filename

    def save_checkpoint(self, metric, criteria, filename: Union[None, str] = None, freq: int = 1):
        """
        Save checkpoint function. By default saves last epoch for training and best validation epoch. If there is
        no validation then best train epoch.

        :param filename: Custom filename to save weights.
        :type filename: [str,None]
        :param freq: Frequency (epoch-wise) with which checkpoint will be saved.
        :return: None
        """
        if self.epoch % freq == 0:
            filename = self._save_checkpoint(filename)
            if getattr(self, metric + '_')[self.state].is_best(criteria):
                shutil.copyfile(filename,
                                os.path.join(self.IO.metadata_dir, '%04d' % self.epoch + '_best_checkpoint.pth'))
            print('Checkpoint saved successfully')

    def checkpoint(self, metric: str = 'loss', criteria=[min, lambda epoch, optimal: optimal > epoch],
                   filename: Union[None, str] = None,
                   freq: int = 1):

        return partial(self.save_checkpoint, metric=metric, criteria=criteria, filename=filename, freq=freq)

    def load_checkpoint(self, metadata_dir: str):
        """
        Load a checkpoint from flerken framework given a experiment path
        :param metadata_dir: Path to the experiment
        :type metadata_dir: str
        :return: Returns the chosen checkpoint from the available candidates in the experiment folder.
        """
        candidates = self._gather_candidates(metadata_dir)
        if len(candidates) == 0:
            raise FileNotFoundError('No checkpoint detected at %s' % metadata_dir)
        directory = os.path.join(self.IO.resume_from, candidates[0])

        print("=> Loading checkpoint '{}'".format(metadata_dir))
        checkpoint = torch.load(directory, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch']
        self._model.load(checkpoint, from_checkpoint=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Detecting scalars...')
        for key in checkpoint['tsi']:
            setattr(self, key + '_', checkpoint['tsi'][key])
            print('>' + key + ' loaded!')
        self.absolute_iter = checkpoint['iter']
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(directory, checkpoint['epoch']))
        return directory

    def _gather_candidates(self, metadata_dir):
        """
        Generate a list of sorted candidates (paths) to be loaded given a metadata_dir path.
        :param metadata_dir: Path to a metadata directory
        :type metadata_dir: str
        :return: list
        """
        candidates = os.listdir(metadata_dir)
        candidates = [x for x in candidates if x.endswith(('.pth', '.pt'))]

        return sorted(candidates)[::-1]


class Experiment(object):
    def __init__(self, arxiv: str, workname: Union[None, str] = None):
        if not isinstance(arxiv, str):
            raise TypeError('input "arxiv" should be of type str but %s passed' % type(arxiv))
        if (not isinstance(workname, str)) and workname != None:
            raise TypeError('input "workname" should be of type str or None but %s passed' % type(arxiv))
        self.arxiv = Arxiv(arxiv)
        if workname is None:
            workname = str(datetime.datetime.now())[:19]
        self.workname = workname
        self.IO = FileManager(os.path.join(self.arxiv.dir, self.workname))

    def __repr__(self):
        return "Flerken Ex: Experiment name: %s Directory: %s" % (self.workname, self.IO.workdir)

    def __getattr__(self, item):
        try:
            return self.cfg_files[item]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(
                type(self).__name__, item))

    # SHORTCUTS as @property
    @property
    def resume_from(self) -> Union[None, str]:
        return self.IO.resume_from

    @property
    def cfg_files(self):
        return self.IO.cfg_files

    def _autoconfig_scratch(self, trainer: Trainer) -> Trainer:
        print('Autoconfig trainer...')
        if not hasattr(trainer, 'EPOCHS'):
            trainer.EPOCHS = 50
        if not hasattr(trainer, 'optimizer'):
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        if not hasattr(trainer, 'criterion'):
            trainer.criterion = torch.nn.L1Loss().to(trainer.alloc.main_device)
        if not hasattr(trainer, 'scheduler'):
            trainer.scheduler = None
        trainer.IO = self.IO
        return trainer

    def _autoconfig_resume(self, trainer: Trainer) -> Trainer:
        return self._autoconfig_scratch(trainer)

    def autoconfig(self, trainer: Trainer) -> Trainer:
        if self.resume_from is not None:
            trainer = self._autoconfig_resume(trainer)
        else:
            trainer = self._autoconfig_scratch(trainer)
        return trainer
