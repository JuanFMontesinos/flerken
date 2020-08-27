import os
import sys
import datetime
import subprocess
import random
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ..utils import BaseDict, ClassDict
from . import sqlite_tools

__all__ = ['FileManager']


def experiment_cfg():
    return {
        "metadata_foldername": 'metadata',
        "seed_filename": '.seed',
        "tracked_cfg_files_filename": '.tracked_cfg.json',
        "internal_cfg_filename": ".internal_cfg.json",
        "internal_cfg_dicttype": ClassDict,
        "metrics_dicttype": BaseDict,
        "cfg_files_dicttype": ClassDict}


def arxiv_cfg():
    return {
        "database_name": 'experiment_db.sqlite'
    }


class FileManager(object):
    def __init__(self, workdir):
        self.cfg_files = experiment_cfg()["cfg_files_dicttype"]()
        self.workdir = workdir
        self._enter_called = False
        self.packages = []
        if self.exists():
            self.resume_from = self._resume()
        else:
            self._init_files()
            self.resume_from = None
        self._write_internal_cfg()

    @property
    def metadata_dir(self):
        return getattr(self, '_metadata_dir', None)

    @metadata_dir.setter
    def metadata_dir(self, path):
        assert os.path.isdir(path)
        self._metadata_dir = path

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, path):
        assert isinstance(path, str)
        print('Setting workdir at {}'.format(path))
        self._workdir = path

    def add_cfg(self, key, dictionary, overwrite=False):
        if self._enter_called:
            raise PermissionError('Cfg files should be added before using the context manager.')
        if not isinstance(dictionary, dict):
            raise TypeError('add_cfg requires a dictionary as input')

        dictionary = experiment_cfg()["internal_cfg_dicttype"](dictionary)
        if overwrite:
            self.cfg_files[key] = experiment_cfg()["internal_cfg_dicttype"]({key: dictionary})
        else:
            self.cfg_files = self.cfg_files + experiment_cfg()["internal_cfg_dicttype"]({key: dictionary})

    def add_package(self, obj):
        self.packages.append(obj.__path__[0])

    def exists(self):
        return os.path.exists(self.workdir)

    def write_cfg(self):
        metadata_dir = os.path.join(self.workdir,
                                    experiment_cfg()["metadata_foldername"],
                                    str(self._internal_cfg['version']))
        os.makedirs(metadata_dir)
        self.metadata_dir = metadata_dir
        if self.cfg_files.get(experiment_cfg()['seed_filename']) is None:
            self.add_cfg(experiment_cfg()['seed_filename'],
                         experiment_cfg()["internal_cfg_dicttype"]
                         ({'seed': self.seed}))
        # self.set_seed(self.seed)
        for key in self.cfg_files:
            self.cfg_files[key].write(os.path.join(self.metadata_dir, key))
        with open(os.path.join(self.metadata_dir, "sysinfo.txt"), 'w') as file:
            file.write(self.get_sys_info())

        for src in self.packages:
            dst = src.replace(os.path.dirname(src), os.path.join(self.metadata_dir, 'packages'))
            shutil.copytree(src, dst)
        return self.metadata_dir

    def cfg_from(self, workdir):
        if not os.path.exists(workdir):
            raise ValueError('Autoconfig_from path %s does not exist for the given arxiv' % workdir)
        json_path = os.path.join(workdir, experiment_cfg()['internal_cfg_filename'])
        _internal_cfg = experiment_cfg()['internal_cfg_dicttype']().load(json_path)
        metadata_dir = os.path.join(workdir,
                                    experiment_cfg()["metadata_foldername"],
                                    str(_internal_cfg['version']))
        files, names = self.load_cfg(metadata_dir)
        for file, name in zip(files, names):
            self.add_cfg(name, file)
        print('Configuration files loaded from %s: ' % metadata_dir)
        return metadata_dir

    @staticmethod
    def load_cfg(metadata_dir):
        assert os.path.exists(metadata_dir)
        cfg_files = [x for x in os.listdir(metadata_dir) if x.endswith('.json')]
        files = []
        names = []
        for name_ex in cfg_files:
            path = os.path.join(metadata_dir, name_ex)
            name = os.path.splitext(name_ex)[0]
            file = experiment_cfg()["internal_cfg_dicttype"]().load(path)
            # self.add_cfg(name, file)
            files.append(file)
            names.append(name)
        return files, names

    @staticmethod
    def get_sys_info():
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

        return ('\r\t __Python VERSION: {0} \r\t'
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

    @property
    def seed(self):
        if not hasattr(self, '_seed'):
            random.seed(None)
            self._seed = random.randint(0, 2 ** 32 - 1)

        return self._seed

    @seed.setter
    def seed(self, val):
        assert isinstance(val, int)
        assert 0 <= val <= sys.maxsize
        self._seed = val

    @staticmethod
    def set_seed(value):
        torch.manual_seed(value)
        np.random.seed(value)
        random.seed(value)

    def _init_files(self):
        os.makedirs(self.workdir)
        os.makedirs(os.path.join(self.workdir, experiment_cfg()["metadata_foldername"]))
        self._internal_cfg = self._create_internal_cfg()

    def _resume(self):
        self._internal_cfg = experiment_cfg()["internal_cfg_dicttype"]()
        self._internal_cfg.load(os.path.join(self.workdir, experiment_cfg()["internal_cfg_filename"]))
        metadata_dir = self.cfg_from(self.workdir)
        self._internal_cfg = self._update_internal_cfg(self._internal_cfg)
        return metadata_dir

    @staticmethod
    def _create_internal_cfg():
        internal_cfg = experiment_cfg()["internal_cfg_dicttype"]()
        internal_cfg['version'] = 0
        internal_cfg['creation_date'] = str(datetime.datetime.now())
        internal_cfg['lastmod_date'] = internal_cfg['creation_date']
        internal_cfg['modification_date_array'] = []
        internal_cfg['modification_date_array'].append(internal_cfg['creation_date'])
        return internal_cfg

    def _write_internal_cfg(self):
        self._internal_cfg.write(os.path.join(self.workdir, experiment_cfg()["internal_cfg_filename"]))

    @staticmethod
    def _update_internal_cfg(internal_cfg):
        internal_cfg['version'] += 1
        internal_cfg['lastmod_date'] = str(datetime.datetime.now())
        internal_cfg['modification_date_array'].append(internal_cfg['lastmod_date'])
        return internal_cfg

    def _set_writer(self, **kwargs):
        if not kwargs:
            kwargs = {'log_dir': os.path.join(self.workdir, 'tensorboard')}
            self.summary_writer_path = kwargs['log_dir']

        self.writer = SummaryWriter(**kwargs)


class Arxiv(object):
    def __init__(self, dir):
        self.dir = dir
        if self.exists():
            self._resume()
        else:
            self._init_files()

        self.db_name = 'experiment_db.sqlite'

    def exists(self):
        return os.path.exists(self.dir)

    def _init_files(self):
        print("Creating arxiv at: %s " % self.dir)
        os.makedirs(self.dir)
        self.db_dir = os.path.join(self.dir, arxiv_cfg()["database_name"])
        self.db = sqlite_tools.sq(self.db_dir)

    def _resume(self):
        print("Loading arxiv from: %s " % self.dir)
        self.db_dir = os.path.join(self.dir, arxiv_cfg()["database_name"])
        self.db = sqlite_tools.sq(self.db_dir)
