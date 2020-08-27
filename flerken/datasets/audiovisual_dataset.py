import torch
import numpy as np
from torchtree import Directory_Tree
from torch.utils.data import Dataset, SequentialSampler
from flerken.utils import BaseDict, get_transforms, Processor

import os
from warnings import warn
from random import randint, random, choice
from itertools import zip_longest
import bisect

__all__ = ['AVDataset', 'FileManager', ]


class AudioReader(object):
    def __init__(self):
        self.allowed_formats = ('wav', 'default')
        self._init_formats()

    def _init_formats(self):
        self.formats = BaseDict()
        for format in self.allowed_formats:
            getattr(self, '_' + format)()

    def _scipy(self):
        from scipy.io.wavfile import read
        def reader(x, offset, length):
            if length is not None:
                audio = read(x)[1][offset:offset + length]
            else:
                audio = read(x)[1][offset:]
            return audio

        return reader

    def _sounfile(self):
        from soundfile import read
        def reader(x, offset, length):
            if length is None:
                stop = -1
            else:
                stop = offset + length
            audio, sr = read(x, dtype='int16', start=offset, stop=stop)
            return audio

        return reader

    def _wav(self):
        try:
            self.formats['wav'] = self._sounfile()
        except ImportError:
            self.formats['wav'] = self._scipy()
        except ImportError:
            print('Supported audio libraries are soundfile and scipy.io wavfile')

    def _default(self):
        try:
            self.formats['default'] = self._sounfile()
        except ImportError:
            self.formats['default'] = self._scipy()
        except ImportError:
            print('Supported audio libraries are soundfile and scipy.io wavfile')

    def __call__(self, path, offset=0, length=None):
        ext = os.path.splitext(path)[1][1:]

        f = self.formats.get(ext, self.formats['default'])
        if ext == '' and os.path.isdir(path):
            files = [os.path.join(path, x) for x in os.listdir(path)]
            if len(files) == 1:
                return f(files[0], offset=offset, length=length)
            elif len(files) == 0:
                raise FileExistsError('Audio resource of format directory is empty: %s' % path)
            else:
                return [f(x, offset=offset, length=length) for x in files]
        else:
            return f(path, offset=offset, length=length)


class ImageReader(object):
    def __init__(self, library='PIL'):
        self.allowed_formats = ('jpg', 'jpeg', 'png', 'default')
        self.f = getattr(self, '_' + library)()

    def _imageio(self):
        from imageio import imread
        def reader(x):
            im = imread(x)
            return im.transpose(2, 0, 1)

        return reader

    def _cv2(self):
        from cv2 import imread, cvtColor, COLOR_BGR2RGB
        def reader(x):
            im = imread(x)
            return cvtColor(im, COLOR_BGR2RGB)

        return reader

    def _PIL(self):
        from PIL import Image
        def reader(x):
            im = Image.open(x)
            return im

        return reader

    def __call__(self, path, **kwargs):
        return self.f(path, **kwargs)


class HasableTree(Directory_Tree):
    def __init__(self, path=None, ignore: list = [], scan_params=True):
        super(HasableTree, self).__init__(path, ignore, scan_params)
        self._tree_properties['path'] = path
        self._tree_properties['hashable'] = {}
        for path in self.paths(self._tree_properties['path']):
            json = BaseDict().load(path)
            id_ = path.split('/')[-1].split('.')[0]
            self._tree_properties['hashable'].update({id_: json})

    def __getitem__(self, item):
        return self._tree_properties['hashable'][item]

    def keys(self):
        return self._tree_properties['hashable'].keys()

    def items(self):
        return self._tree_properties['hashable'].items()

    def values(self):
        return self._tree_properties['hashable'].values()

    def dic(self):
        return self._tree_properties['hashable']


class FileManager(object):
    def __init__(self, root, in_memory=False, as_generator=True, exclude=[],yield_mode='yield_module'):

        self.yield_mode = yield_mode
        if not os.path.exists(root):
            raise Exception('Directory %s does not exist' % root)
        self.root = root
        self.in_memory = in_memory #TODO This won't probably work as generator due to the ordering of scandir
        #which reads files in a not sorted way. Difficult to fix

        self.tree = Directory_Tree(path=root, ignore=exclude, scan_params=self.in_memory)
        self.is_info = 'info' in self._available_resources
        self.len_called = False
        if self.is_info:
            self.info = HasableTree(path=os.path.join(root, 'info'), ignore=exclude, scan_params=True)
        else:
            warn('Info resource not found at %s. Info is necessary in order to save RAM' % root)

        print('Following resources found:')
        for folder in self._available_resources:
            print('\t %s' % folder)
        print('Following resources will be ignored')
        for r in exclude:
            print('\t %s' % r)
        self.resources = [x for x in self._available_resources if x not in exclude]
        self._assert_paired_resources()  # Checks all modules are the same
        self.as_generator = as_generator

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def as_generator(self):
        return self._as_generator

    @as_generator.setter
    def as_generator(self, val):
        self._as_generator = val
        if val == True:
            if hasattr(self, 'indexable'):
                del self.indexable
        elif val == False:
            self.indexable = tuple(iter(self))
        else:
            raise TypeError('as_generator is a binary variable but %s passed' % type(val))

    @property
    def _available_resources(self):
        return tuple(x[0] for x in self.tree.named_children())

    @_available_resources.setter
    def _available_resources(self, val):
        raise EnvironmentError('Available resources emerge from tree, cannot be overwritten by hand')

    def _assert_paired_resources(self):
        files = []
        for name, module in self.tree.named_children():
            files.append(tuple(x[0] for x in module.named_children()))
        pattern = files[0]
        assert all([f == pattern for f in files])

    def __len__(self):
        if not self.len_called:
            self.clusters, self.N = self._calc_len()
            self.len_called = True
        return self.N

    def _calc_len(self):
        # TODO this should be adapted for yield_file?
        class_size = {}
        idx_prev = 0
        if self.is_info:
            for name, module in self.info.named_children():
                n = len(tuple(module.paths()))
                class_size[name] = (idx_prev, idx_prev + n)
                idx_prev += n

        else:
            for name, module in self.tree(self.resources[0]).named_children():
                n = len(tuple(module.named_children()))
                class_size[name] = (idx_prev, idx_prev + n)
                idx_prev += n
        return class_size, idx_prev

    def yield_module(self, obj, resource, root):
        for name, module in obj(resource).named_modules(prefix=os.path.join(root, resource)):
            if module.level() == 3:
                if not self.in_memory:
                    yield (name, module)
                else:
                    yield name
    def yield_file(self, obj, resource, root):
        for path in sorted(list(obj(resource).paths(root=os.path.join(root, resource)))):
            yield path

    def __iter__(self):
        if self.yield_mode == 'yield_module':
            return zip_longest(*[self.yield_module(self.tree, resource, self.root) for resource in self.resources])
        elif self.yield_mode == 'yield_file':
            return zip_longest(*[self.yield_file(self.tree, resource, self.root) for resource in self.resources])
    def __getitem__(self, idx):
        if self.as_generator:
            for i, sample in enumerate(iter(self)):
                if i == idx:
                    return sample
        else:
            return self.indexable[idx]


class AVDataset(Dataset):
    def __init__(self, root, in_memory=False, as_generator=True, exclude=[], debug=False,yield_mode='yield_module', **kwargs):
        self.filemanager = FileManager(root, in_memory=in_memory, as_generator=as_generator, exclude=exclude,yield_mode=yield_mode)
        self.processor = get_transforms()
        self.reader = Reader(self.filemanager.resources, debug=debug, **kwargs)
        self.debug = debug

    def __len__(self):
        return len(self.filemanager)

    def get_idx_kwargs(self, idx):
        raise NotImplementedError

    def sample_idx(self, idx):
        raise NotImplementedError

    def index_kw_sampler(self, N, classes):
        for i in range(N):
            if classes is None:
                idx = randint(0, len(self) - 1)
            else:
                idx_min, idx_max = self.filemanager.clusters[choice(classes)]
                idx = randint(idx_min, idx_max - 1)
            yield idx, self.sample_idx(idx)

    def getitem(self, idx, N, elements, classes=None, trazability=False, **kwargs):
        if trazability:
            trace = {'indices': [], 'kwargs': []}

        if isinstance(idx, int):
            kw = self.sample_idx(idx)
            if self.debug:
                print('Sample %d:' % idx)
                for path, resource in zip(self.filemanager[idx], self.filemanager.resources):
                    if path is not None:
                        key = path.split('/')[-1]
                    print('\t Resource: %s, Path: %s' % (resource, path))
                print('\t Kwargs: %s ' % kw)
                print('\t Info: %s ' % self.filemanager.info[os.path.splitext(key)[0]])
            files = self.reader(self.filemanager[idx], elements, **kw)
            trace = {'indices': [idx], 'kwargs': [kw]}
            obj = self.index_kw_sampler(N - 1, classes)
        elif idx is None:
            obj = self.index_kw_sampler(N, classes)
            files = [[] for _ in elements]
        else:
            obj = idx
            files = [[] for _ in elements]

        for i, (idx, kw) in enumerate(obj):
            if self.debug:
                print('Sample %d:' % idx)
                for path, resource in zip(self.filemanager[idx], self.filemanager.resources):
                    if path is not None:
                        key = path.split('/')[-1]
                    print('\t Resource: %s, Path: %s' % (resource, path))
                print('\t Kwargs: %s ' % kw)
                print('\t Info: %s ' % self.filemanager.info[os.path.splitext(key)[0]])
            if trazability:
                trace['indices'].append(idx)
                trace['kwargs'].append(kw)

            for j, f in enumerate(self.reader(self.filemanager[idx], elements, **kw)):
                files[j].append(f[0])
        if trazability:
            return files, trace
        else:
            return files


class Reader(object):
    def __init__(self, resources, debug):
        self.resources = {x: i for i, x in enumerate(resources)}
        self.debug = debug
        self.init_reader()

    def init_reader(self, **kwargs):
        self.functional = {}
        self.functional['audio'] = AudioReader()
        self.functional['frames'] = ImageReader()
        self.functional.update(kwargs)

    def __getitem__(self, item):
        return self.functional[item]

    def __call__(self, paths, elements, **kwargs):
        if self.debug:
            assert len(self.resources) == len(paths)
        return [[self.functional[el](paths[self.resources[el]], **kwargs[el])] for el in elements]
