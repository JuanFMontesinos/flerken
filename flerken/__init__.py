#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from .framework.pytorchframework import pytorchfw

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