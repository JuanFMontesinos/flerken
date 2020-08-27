from functools import partial
from collections import OrderedDict
from typing import Union

from torchsummary import summary
from torch import nn, load, device

try:
    SUMMARY = True
except ImportError:
    SUMMARY = False

from . import network_initialization

from torch.utils.data import TensorDataset


class Model(object):
    def __init__(self, model: nn.Module, input_shape=None, initializer='Xavier'):
        self.model = model
        self.shape = input_shape
        self.initializable_layers = [self.model.parameters()]
        self.initializer = initializer
        self.hooks = {}
        self._hook_handlers = {}

    def _initialize_layers(self):
        if self.initializable_layers is None:
            print('Network not automatically initilized')
            return False
        else:
            print('Network automatically initilized at : \n ')
            for m in self.initializable_layers:
                print('\t' + m.__class__.__name__)
            map(self.init_function, self.initializable_layers)
            return True

    def with_summary(self):
        if self.shape is not None and SUMMARY:
            return True
        else:
            return False

    def get_summary(self, device: str = 'cuda'):
        """
        Obtain a model sumary using torch.summary.
        Otherwise it will just return model's __repr__

        :param device: device in which to test the summary
        :type str:
        :return: summary
        """
        if self.with_summary():
            return summary(self.model, self.shape, device=device)
        else:
            return str(self.model)

    def init_function(self):
        """
        :return: Returns a function which initialize layers given an iterator of parameters.
        """
        return partial(network_initialization.init_weights, init_type=self.initializer)

    def load(self, directory, strict_loading=True, from_checkpoint=False):
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
            state_dict = load(directory, map_location=lambda storage, loc: storage)
        if from_checkpoint:
            state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict, strict=strict_loading)

    def get_dataset(self, N=2, gt=None, visualization=None):
        if self.shape is None:
            raise ValueError('get_dataset method called but no input shape provided')
        from torch import rand
        if isinstance(self.shape, tuple):
            input_size = [self.shape]
        x = []
        if gt is not None:
            x.append(rand(N, *gt))
        x.append(rand(N, *input_size[0]).float())

        if visualization is not None:
            x.append(rand(N, *visualization))
        return TensorDataset(*x)

    def to(self, device: Union[str, device]):
        return Model(self.model.to(device), self.shape, self.initializer)

    def set_forward_hook(self, name: str, layer: nn.Module, fn = None):
        """
        Set a forward hook which stores the input and output for a given layer.
        Check https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()

        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_forward_hook(self._fwd_hook_fn)
        self._hook_handlers[name] = {'type': 'forward_hook', 'handler': handler, 'fn': fn}
        layer._flerken_fwd_hook_name = name

        return handler

    def set_backward_hook(self, name: str, layer: nn.Module, fn= None):
        """
        Set a backward hook which stores the input gradient and output gradient for a given layer.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()
        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_backward_hook(self._bwd_hook_fn)
        self._hook_handlers[name] = {'type': 'backward_hook', 'handler': handler, 'fn': fn}
        layer._flerken_bwd_hook_name = name

        return handler

    def set_forward_pre_hook(self, name: str, layer: nn.Module, fn= None):
        """
        Set a pre-hook (which is like a forward hook but called before calling the layer)
         which stores the input for a given layer.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        for additional info.

        The funtion fn should be written as official pytorch docs state.

        :param name: Unique name for the hook.
        :type name: str
        :param layer: module to be hooked
        :type layer: nn.Module
        :param fn: Optional function to modify the hook.
        :param fn: Callable
        :return: torch.utils.hooks.RemovableHandle
        a handle that can be used to remove the added hook by calling handle.remove()
        """
        if name in self._hook_handlers:
            raise ValueError(f'Hook name {name} already in use')

        handler = layer.register_forward_pre_hook(self._fwd_hook_pre_fn)
        self._hook_handlers[name] = {'type': 'forward_pre_hook', 'handler': handler, 'fn': fn}
        layer._flerken_fwd_pre_hook_name = name

        return handler

    def del_hook(self, name):
        self._hook_handlers[name]['handler'].remove()
        self._hook_handlers.pop(name)

    def _fwd_hook_fn(self, module, input, output):
        name = module._flerken_fwd_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, input, output = self._hook_handlers[name]['fn'](module, input, output)
        self.hooks[name] = {'input': input, 'output': output}

    def _fwd_hook_pre_fn(self, module, input):
        name = module._flerken_fwd_pre_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, input = self._hook_handlers[name]['fn'](module, input)
        self.hooks[name] = {'input': input}

    def _bwd_hook_fn(self, module, grad_in, grad_out):
        name = module._flerken_bwd_hook_name
        if self._hook_handlers[name]['fn'] is not None:
            module, grad_in, grad_out = self._hook_handlers[name]['fn'](module, grad_in, grad_out)
        self.hooks[name] = {'input': grad_in, 'output': grad_out}
        if self._hook_handlers[name]['fn'] is not None:
            return  grad_in
