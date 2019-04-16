__author__ = "Juan Montesinos"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import torch as _torch
from .framework.pytorchframework import pytorchfw
from . import datasets
from . import dataloaders

def cuda_timing(func):
    def inner(*args,**kwargs):
        start = _torch.cuda.Event(enable_timing=True)
        end = _torch.cuda.Event(enable_timing=True)
        start.record()
        output = func(*args,**kwargs)
        end.record()
        _torch.cuda.synchronize()
        time = start.elapsed_time(end)
        return output,time
    return inner