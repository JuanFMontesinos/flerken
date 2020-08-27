from . import toy_fw, toy_example
from flerken.framework.framework import Experiment, Trainer
from flerken.framework.experiment import FileManager, Arxiv, experiment_cfg
from flerken.framework import *
from flerken.framework.allocator import Allocator
from flerken.framework.model import Model
from flerken.framework.debug import NaNError, InfError

import torch
from torch.utils.data import DataLoader
import unittest

import shutil
import os


class TestFramework(unittest.TestCase):
    def setUp(self) -> None:
        self.arxiv_path = './arxiv'
        if os.path.exists(self.arxiv_path):
            shutil.rmtree(self.arxiv_path)
        self.workdir = os.path.join(self.arxiv_path, 'canelon')
        self.ex = Experiment(self.arxiv_path, 'canelon')

    def tearDown(self) -> None:
        if os.path.exists(self.arxiv_path):
            shutil.rmtree(self.arxiv_path)

    def test_init(self):
        trainer = Trainer('cuda:0', toy_example())

    def test_autoconfig_scratch(self):
        from torch.optim import optimizer
        trainer = Trainer('cuda:0', toy_example())
        trainer = self.ex.autoconfig(trainer)
        self.assertEqual(self.ex.resume_from, None)
        self.assertTrue(isinstance(trainer.EPOCHS, int))
        self.assertTrue(isinstance(trainer.optimizer, optimizer.Optimizer))
        self.assertTrue(hasattr(trainer, 'criterion'))
        self.assertTrue(hasattr(trainer, 'scheduler'))

    def test_cfg_from(self):
        trainer = Trainer('cuda:0', toy_example(), input_shape=(1, 15, 15))
        json = experiment_cfg()['internal_cfg_dicttype']({'pill': 1, 'q': 5})
        json2 = experiment_cfg()['internal_cfg_dicttype']({'pez': 2, 'casa': []})
        self.ex.IO.add_cfg('test1', json)
        self.ex.IO.add_cfg('test2', json2)
        with self.ex.autoconfig(trainer) as trainer:
            pass
        ex = Experiment(self.arxiv_path, 'croqueta')
        ex.IO.cfg_from(os.path.join(ex.arxiv.dir, 'canelon'))
        with ex.autoconfig(trainer) as trainer:
            self.assertEqual(ex.IO.cfg_files.keys(), self.ex.IO.cfg_files.keys())

    def test_enter(self):
        trainer = Trainer('cuda:0', toy_example(), input_shape=(1, 15, 15))
        json = experiment_cfg()['internal_cfg_dicttype']({'pill': 1, 'q': 5})
        json2 = experiment_cfg()['internal_cfg_dicttype']({2: 2, 3: []})
        self.ex.IO.add_cfg('test1', json)
        self.ex.IO.add_cfg('test2', json2)
        with self.ex.autoconfig(trainer) as trainer:
            self.assertEqual(trainer.IO.metadata_dir, os.path.join(self.workdir, 'metadata', '0'))
            self.assertEqual(trainer.IO.workdir, self.workdir)
            model_path = os.path.join(trainer.IO.metadata_dir, 'model.txt')
            test1_path = os.path.join(trainer.IO.metadata_dir, 'test1.json')
            test2_path = os.path.join(trainer.IO.metadata_dir, 'test2.json')
            internal_cfg_path = os.path.join(trainer.IO.workdir, experiment_cfg()['internal_cfg_filename'])
            sys_path = os.path.join(trainer.IO.metadata_dir, 'system_info.txt')
            tb_path = os.path.join(trainer.IO.workdir, 'tensorboard')
            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(sys_path))
            self.assertTrue(os.path.exists(tb_path))
            self.assertTrue(os.path.exists(test1_path))
            self.assertTrue(os.path.exists(test2_path))
            self.assertTrue(os.path.exists(internal_cfg_path))

    def test_set_model_training(self):
        trainer = Trainer('cuda:0', toy_example(), input_shape=(1, 15, 15))
        flag = trainer.set_model_training(True)
        self.assertTrue(flag)
        self.assertTrue(trainer.model.training)
        flag = trainer.set_model_training(False)
        self.assertFalse(flag)
        self.assertFalse(trainer.model.training)

    def test_train(self):
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))

        dataloader = DataLoader(trainer._model.get_dataset(2, gt=(10, 11, 11), visualization=(1,)), batch_size=1)
        with self.ex.autoconfig(trainer) as trainer:
            for trainer.epoch in range(trainer.EPOCHS):
                trainer.run_epoch(dataloader, 'train', checkpoint=trainer.checkpoint(), backprop=True, debug=True)
        paths = [os.path.join(trainer.IO.metadata_dir, '%04d' % x + '_checkpoint.pth') for x in range(trainer.EPOCHS)]
        for path in paths:
            self.assertTrue(os.path.exists(path))

    def test_add_package(self):
        import flerken
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))
        self.ex.IO.add_package(flerken)
        with self.ex.autoconfig(trainer) as trainer:
            self.assertTrue(os.path.exists(os.path.join(trainer.IO.metadata_dir, 'packages', 'flerken')))

    def test_keyboard_interrupt_checkpoint(self):
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))

        dataloader = DataLoader(trainer._model.get_dataset(2, gt=(10, 11, 11), visualization=(1,)), batch_size=1)
        with self.assertRaises(KeyboardInterrupt), self.ex.autoconfig(trainer) as trainer:
            for trainer.epoch in range(trainer.EPOCHS):
                trainer.run_epoch(dataloader, 'train', checkpoint=trainer.checkpoint(), backprop=True, debug=True)
                if trainer.epoch == 5:
                    raise KeyboardInterrupt()
                    self.assertTrue(
                        os.path.exists(os.path.join(trainer.IO.metadata_dir, 'KeyboardInterrupt_checkpoint.pth')))

    def test_freq_checkpoint(self):
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))

        dataloader = DataLoader(trainer._model.get_dataset(2, gt=(10, 11, 11), visualization=(1,)), batch_size=1)
        with self.ex.autoconfig(trainer) as trainer:
            for trainer.epoch in range(trainer.EPOCHS):
                trainer.run_epoch(dataloader, 'train', checkpoint=trainer.checkpoint(freq=2), backprop=True, debug=True)
                if (trainer.epoch % 2) == 0:
                    self.assertTrue(
                        os.path.exists(
                            os.path.join(trainer.IO.metadata_dir, '%04d' % trainer.epoch + '_checkpoint.pth')))

    def test_autoconfig_resume(self):
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))

        dataloader = DataLoader(trainer._model.get_dataset(2, gt=(10, 11, 11), visualization=(1,)), batch_size=1)
        json = experiment_cfg()["internal_cfg_dicttype"]({'jamon': 7})
        self.ex.IO.add_cfg('test1', json)
        with self.ex.autoconfig(trainer) as trainer:
            for trainer.epoch in range(trainer.start_epoch, trainer.EPOCHS // 2):
                trainer.run_epoch(dataloader, 'train', checkpoint=trainer.checkpoint(), backprop=True, debug=True)
        self.assertEqual(self.ex.IO.cfg_files['test1'], json)
        sd = trainer.model.state_dict().copy()
        ex = Experiment(self.arxiv_path, 'canelon')
        with ex.autoconfig(trainer) as trainer:
            self.assertEqual(self.ex.IO._internal_cfg['version'] + 1, ex.IO._internal_cfg['version'])
            # Same weights
            for key in sd:
                self.assertTrue((trainer.model.state_dict()[key] == sd[key]).all())
            self.assertEqual(ex.IO.cfg_files['test1'], json)
            self.assertEqual(trainer.epoch + 1, trainer.start_epoch)

            for trainer.epoch in range(trainer.start_epoch, trainer.EPOCHS):
                trainer.run_epoch(dataloader, 'train', checkpoint=trainer.checkpoint(), backprop=True, debug=True)

    def test_add_cfg_ctx_manager(self):
        if torch.cuda.is_available():
            trainer = Trainer('cuda:0', toy_example().cuda(), input_shape=(1, 15, 15))
        else:
            trainer = Trainer('cpu', toy_example(), input_shape=(1, 15, 15))
        json = experiment_cfg()["internal_cfg_dicttype"]({'jamon': 7})
        with self.assertRaises(PermissionError), self.ex.autoconfig(trainer):
            self.ex.IO.add_cfg('test1', json)


class TestFrameworkOld(unittest.TestCase):
    def setUp(self) -> None:
        self.fw = toy_fw(toy_example(), './', None, 'cpu', False)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.fw.workdir)
        except:
            pass

    def test_init(self):
        fw = toy_fw(toy_example(), './', None, 'cpu', False)
        self.assertIsInstance(fw.inizilizable_layers, list)
        self.assertFalse(fw.iterating)
        self.assertFalse(fw.loaded_model)
        self.assertIsNone(fw.scheduler)
        self.assertEqual(fw.prevstate, 'train')
        self.assertEqual(fw.assertion_variables,
                         ['initializer', 'EPOCHS', 'optimizer', 'criterion', 'LR', 'dataparallel'])
        self.assertIsInstance(fw.main_device, torch.device)
        self.assertFalse(fw.workdir_enabled)
        self.assertEqual(fw.state, 'train')

    def test_train_and_resume(self):
        from numbers import Number
        from flerken.framework import classitems
        self.fw = toy_fw(toy_example(), './', None, 'cpu', True, debug=False)
        self.fw.train()
        self.assertIsInstance(self.fw.loss, float)
        self.assertIsInstance(self.fw.key['ACC'], Number)
        self.assertIsInstance(self.fw.key['VACC'], Number)
        key = self.fw.key

        self.fw = toy_fw(toy_example(), './', self.fw.workname, 'cpu', True)
        self.fw.hyperparameters()
        self.fw.scheduler = classitems.Scheduler(self.fw.scheduler)
        self.assertTrue(self.fw.resume)
        self.fw.EPOCHS = 2
        self.fw._loadcheckpoint()
        self.assertEqual(self.fw.start_epoch, 1)
        self.maxDiff = None
        for x in self.fw.key:  # Key 2 key comparison cannot be done as values are added after saving checkpoint
            if x == 'ITERATIONS':
                self.assertEqual(self.fw.key[x], key[x] + 1)
            else:
                self.assertEqual(self.fw.key[x], key[x])
        self.fw = toy_fw(toy_example(), './', self.fw.workname, 'cpu', True)
        self.fw.train()

    def test_train_debugger_normal(self):
        fw = toy_fw(toy_example(isnan=True), './', None, 'cpu', True, debug=True)
        with self.assertRaises(NaNError):
            fw.train()

    def test_set_model_training(self):
        flag = self.fw.set_model_training(True)
        self.assertTrue(flag)
        self.assertTrue(self.fw.model.training)
        flag = self.fw.set_model_training(False)
        self.assertFalse(flag)
        self.assertFalse(self.fw.model.training)

    def test_load_model(self):
        weights = torch.load('./test/test_weights.pt')
        self.fw.load_model('./test/test_weights.pt')
        self.assertTrue(self.fw.loaded_model)
        dic = self.fw.model.state_dict()
        self.assertEqual(list(dic.keys()), list(weights.keys()))
        for key in dic.keys():
            assert torch.all(torch.eq(dic[key], weights[key]))

        self.setUp()
        self.fw.load_model(weights)
        self.assertTrue(self.fw.loaded_model)
        self.assertEqual(list(dic.keys()), list(weights.keys()))
        for key in dic.keys():
            assert torch.all(torch.eq(dic[key], weights[key]))

    def test_init_function(self):
        from functools import partial
        from flerken.framework import network_initialization
        types = ('normal', 'xavier', 'kaiming', 'orthogonal')

        for x in types:
            self.fw.init_function = partial(network_initialization.init_weights, init_type=x)
            self.fw.__initialize_layers__()

    def test_epoch(self):
        self.fw.epoch = 7
        self.assertEqual(self.fw.epoch, 7)
        self.assertEqual(self.fw._epoch, 7)
        self.assertEqual(self.fw.key['EPOCH'], 7)

    def test_absolute_iter(self):
        self.fw.absolute_iter = 7
        self.assertEqual(self.fw.absolute_iter, 7)
        self.assertEqual(self.fw._absolute_iter, 7)
        self.assertEqual(self.fw.key['ITERATIONS'], 7)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = toy_example()

    def test_init(self):
        model = Model(self.model)

    def test_with_summary(self):
        model = Model(self.model, input_shape=None)
        self.assertFalse(model.with_summary())
        model = Model(self.model, input_shape=(1, 1, 15, 15))
        self.assertTrue(model.with_summary())

    def test_get_summary(self):
        model = Model(self.model, input_shape=None)
        self.assertEqual(str(self.model), model.get_summary('cuda'))
        model = Model(self.model, input_shape=(1, 15, 15))
        self.assertNotEqual(str(self.model), model.get_summary('cpu'))

    def test_init_layers(self):
        for init_type in ['normal', 'xavier', 'kaiming', 'orthogonal']:
            model = Model(self.model, input_shape=(1, 15, 15), initializer=init_type)
            self.assertTrue(model._initialize_layers())
        model = Model(self.model, input_shape=(1, 15, 15), initializer=init_type)
        model.initializable_layers = None
        self.assertFalse(model._initialize_layers())


class TestAllocator(unittest.TestCase):
    def setUp(self) -> None:
        self.gpu = torch.cuda.is_available()

    def test_item(self):
        alloc = Allocator('cpu', dataparallel=False)
        tensor = alloc(torch.rand(1, 1))
        self.assertTrue(tensor.device == torch.device('cpu'))
        if self.gpu:
            alloc = Allocator('cuda:0', dataparallel=False)
            tensor = alloc(torch.rand(1, 1))
            self.assertTrue(tensor.device == torch.device('cuda:0'))

    def test_list(self):
        alloc = Allocator('cpu', dataparallel=False)
        tensor = alloc([torch.rand(1, 1) for x in range(2)])
        self.assertTrue(isinstance(tensor, list))
        self.assertTrue(all([x.device == torch.device('cpu') for x in tensor]))
        if self.gpu:
            alloc = Allocator('cuda:0', dataparallel=False)
            tensor = alloc([torch.rand(1, 1) for x in range(2)])
            self.assertTrue(isinstance(tensor, list))
            self.assertTrue(all([x.device == torch.device('cuda:0') for x in tensor]))

    def test_tuple(self):
        alloc = Allocator('cpu', dataparallel=False)
        tensor = alloc(tuple([torch.rand(1, 1) for x in range(2)]))
        self.assertTrue(isinstance(tensor, tuple))
        self.assertTrue(all([x.device == torch.device('cpu') for x in tensor]))
        if self.gpu:
            alloc = Allocator('cuda:0', dataparallel=False)
            tensor = alloc(tuple([torch.rand(1, 1) for x in range(2)]))
            self.assertTrue(isinstance(tensor, tuple))
            self.assertTrue(all([x.device == torch.device('cuda:0') for x in tensor]))


class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.model = toy_example()
        self.path = './arxiv'

    def tearDown(self) -> None:
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def test_init_workname_none(self):
        ex = Experiment(self.path)
        self.assertTrue(os.path.exists(ex.IO.workdir))
        self.assertTrue(os.path.exists(ex.arxiv.dir))

    def test_init_workname_named(self):
        ex = Experiment(self.path, workname='ayuso')
        self.assertTrue(os.path.exists(os.path.join(ex.arxiv.dir, 'ayuso')))
        self.assertTrue(os.path.exists(ex.arxiv.dir))

    def test_init_wrong_workname_type(self):
        with self.assertRaises(TypeError):
            ex = Experiment(self.path, workname=5)

    def test_init_wrong_arxiv_type(self):
        with self.assertRaises(TypeError):
            ex = Experiment(16)

    def returns_resume(self):
        ex = Experiment(self.path)
        self.assertTrue(isinstance(ex.resume, bool))
