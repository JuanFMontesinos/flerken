import unittest
from . import toy_fw, toy_example
import torch

import shutil


class TestFramework(unittest.TestCase):
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
        self.fw = toy_fw(toy_example(), './', None, 'cpu', True)
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
        for x in self.fw.key: # Key 2 key comparison cannot be done as values are added after saving checkpoint
            self.assertEqual(self.fw.key[x], key[x])

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
