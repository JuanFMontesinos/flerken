import torch
import sys

__all__ = ['Allocator']


class Allocator(object):
    def __init__(self, main_device, dataparallel: bool):

        try:
            main_device = torch.device(main_device)
        except Exception as ex:
            raise type(ex)(
                str(
                    ex) + "Main:_device should be cpu 'cpu' \
                           or an integer but  %s found" % main_device).with_traceback(
                sys.exc_info()[2])
        self.main_device = main_device
        self.dataparallel = dataparallel

    @property
    def main_device(self):
        return self._main_device

    @main_device.setter
    def main_device(self, value):
        self._main_device = torch.device(value)

    def _allocate(self, x, device):
        if isinstance(x, (list, tuple)):
            for i in x:
                yield i.to(device)
        else:
            yield x.to(device)

    def allocate(self, x, device=None):
        if device is None:
            device = self.main_device
        else:
            device = torch.device(device)
        if self.dataparallel:
            return x
        else:
            if isinstance(x, list):
                return list(self._allocate(x, device))
            elif isinstance(x, tuple):
                return tuple(self._allocate(x, device))
            else:
                return x.to(device)

    def __call__(self, item, device=None):
        return self.allocate(item, device)
