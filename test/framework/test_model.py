import unittest

import torch

from flerken.framework.model import Model


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3),
                                         torch.nn.Conv2d(3, 6, 3))

    def _gen_input(self):
        return torch.rand(1, 1, 30, 30)

    @unittest.skipIf(not torch.cuda.is_available(), 'Cuda not available')
    def allocate_cuda(self):
        model = Model(self.model).to('cuda:0')
        self.assertTrue(model.model.device() == 'cuda:0')

    def test_forward_hook(self):
        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                handler = model.set_forward_hook('test_fwd', i[1])

        self.assertTrue('test_fwd' in model._hook_handlers)
        self.assertTrue(model._hook_handlers['test_fwd']['type'] == 'forward_hook')
        self.assertTrue(model._hook_handlers['test_fwd']['fn'] == None)
        self.assertTrue(isinstance(model._hook_handlers['test_fwd']['handler'], torch.utils.hooks.RemovableHandle))
        self.assertTrue(model._hook_handlers['test_fwd']['handler'] == handler)

        input = self._gen_input()
        gt_result = model.model._modules['0'].forward(input)
        model.model(input)
        self.assertTrue('test_fwd' in model.hooks)
        self.assertTrue((gt_result == model.hooks['test_fwd']['output'][0]).all())

    def test_forward_hook_with_fn(self):

        def hook_fn(module, input, output):
            output[0].zero_()
            return module, input, output

        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                model.set_forward_hook('test_fwd', i[1], fn=hook_fn)
        input = self._gen_input()
        model.model(input)
        self.assertTrue(model._hook_handlers['test_fwd']['fn'] == hook_fn)
        self.assertTrue(
            (model.hooks['test_fwd']['output'][0] == torch.zeros_like(model.hooks['test_fwd']['output'][0])).all())

    def test_backward_hook(self):
        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                handler = model.set_backward_hook('test_bwd', i[1])

        self.assertTrue('test_bwd' in model._hook_handlers)
        self.assertTrue(model._hook_handlers['test_bwd']['type'] == 'backward_hook')
        self.assertTrue(model._hook_handlers['test_bwd']['fn'] == None)
        self.assertTrue(isinstance(model._hook_handlers['test_bwd']['handler'], torch.utils.hooks.RemovableHandle))
        self.assertTrue(model._hook_handlers['test_bwd']['handler'] == handler)

        input = self._gen_input()
        scalar = model.model(input).mean()
        scalar.backward()
        gt_result = model.model._modules['0'].weight.grad
        self.assertTrue('test_bwd' in model.hooks)
        self.assertTrue((gt_result == model.hooks['test_bwd']['input'][1]).all())

    def test_backward_hook_with_fn(self):

        def hook_fn(module, input, output):
            input[1].zero_()
            return module, input, output

        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                handler = model.set_backward_hook('test_bwd', i[1],fn=hook_fn)

        self.assertTrue(model._hook_handlers['test_bwd']['fn'] == hook_fn)

        input = self._gen_input()
        scalar = model.model(input).mean()
        scalar.backward()
        gt_result = torch.zeros_like(model.model._modules['0'].weight.grad)
        self.assertTrue('test_bwd' in model.hooks)
        self.assertTrue((gt_result == model.hooks['test_bwd']['input'][1]).all())

    def test_forward_pre_hook(self):
        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                handler = model.set_forward_pre_hook('test_fwd', i[1])

        self.assertTrue('test_fwd' in model._hook_handlers)
        self.assertTrue(model._hook_handlers['test_fwd']['type'] == 'forward_pre_hook')
        self.assertTrue(model._hook_handlers['test_fwd']['fn'] == None)
        self.assertTrue(isinstance(model._hook_handlers['test_fwd']['handler'], torch.utils.hooks.RemovableHandle))
        self.assertTrue(model._hook_handlers['test_fwd']['handler'] == handler)

        input = self._gen_input()
        model.model(input)
        self.assertTrue('test_fwd' in model.hooks)
        self.assertTrue((input == model.hooks['test_fwd']['input'][0]).all())

    def test_forward_pre_hook_with_fn(self):

        def hook_fn(module, input):
            input[0].zero_()
            return module, input

        model = Model(self.model)
        for i in model.model.named_modules():
            if i[0] == '0':
                model.set_forward_pre_hook('test_fwd', i[1], fn=hook_fn)
        input = self._gen_input()
        model.model(input)
        self.assertTrue(model._hook_handlers['test_fwd']['fn'] == hook_fn)
        self.assertTrue(input.sum()==0)