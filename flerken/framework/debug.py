import torch
from torch import nn
from collections import OrderedDict
from .python_inheritance import ClassDict
from functools import partial

__all__ = ['NaNError', 'InfError', 'Debugger']


class NaNError(Exception):
    pass


class InfError(Exception):
    pass


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class Debugger(nn.Module):
    def __init__(self, model, raise_exception):
        super(Debugger, self).__init__()
        self.model = model
        self.hookF = [Hook(layer[1]) for layer in list(self.model.named_modules())]
        self.hookB = [Hook(layer[1], backward=True) for layer in list(self.model.named_modules())]
        self.hookN = [layer[0] for layer in list(self.model.named_modules())]
        self.raise_exception = raise_exception

    def forward(self, *input, **kwargs):
        return self.model(*input, **kwargs)

    def __call__(self, *args, **kwargs):
        result = super(Debugger, self).__call__(*args, **kwargs)
        nan_dic = self.assertnan(self.hookF, self.hookN, 'fwd')
        inf_dic = self.assertinf(self.hookF, self.hookN, 'fwd')
        if self.isnan:
            self._printsing(nan_dic, 'NaN', 'fwd')
            if self.raise_exception:
                raise NaNError('NaN found while processing this sample.')
            if self.isnan:
                self._printsing(inf_dic, 'Inf', 'fwd')
                if self.raise_exception:
                    raise InfError('NaN found while processing this sample.')
        return result

    def _assertF(self, hook_listF, hook_listN, f, mode):
        assert mode in ['fwd', 'bwd']
        for hookF, name in zip(hook_listF, hook_listN):
            result = OrderedDict()
            if f(hookF.input).any():
                inpF = True
            else:
                inpF = False
            if f(hookF.output).any():
                optF = True
            else:
                optF = False
            if inpF or optF:
                result[name] = {mode: {'inp': inpF, 'opt': optF}}
        return result

    def _printsing(self, dic, value, mode):
        print('Detected %s values:' % value)
        for key in dic:
            result = ClassDict(dic[key][mode])
            print(
                ('Layer: ' + key + '.Input ' + mode + ': {r.inp}. Output ' + mode + ': {r.opt} ').format(r=result))

    @staticmethod
    def check_fault(input, f):
        isnan = False
        if isinstance(input, (list, tuple)):
            for inp in input:
                isnan = isnan or f(inp) if torch.is_tensor(inp) else isnan
        elif torch.is_tensor(input):
            isnan = f(input)
        else:
            raise TypeError('Input type must be torch.tensor, tuple or list')
        return isnan

    def assertnan(self, hook_listF, hook_listN, mode):
        result = self._assertF(hook_listF, hook_listN, partial(self.check_fault, f=torch.isnan), mode=mode)
        self.isnan = bool(result)
        return result

    def assertinf(self, hook_listF, hook_listN, mode):
        result = self._assertF(hook_listF, hook_listN, partial(self.check_fault, f=torch.isinf), mode=mode)
        self.isinf = bool(result)
        return result
