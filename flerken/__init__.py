__author__ = "Juan Montesinos"
__version__ = "0.2.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import torch
from .framework.pytorchframework import pytorchfw
from . import datasets

def cuda_timing(func):
    def inner(*args,**kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = func(*args,**kwargs)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        return output,time
    return inner