#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import transforms
from PIL import Image as _Image
import os as _os
__all__ = ['loadseq']
def loadseq(seq):
    if isinstance(seq,str):
        if _os.path.isdir(seq):
            paths = _os.listdir(seq)
            paths = [_os.path.join(seq,f) for f in sorted(paths)]
            return [_Image.open(f) for f in paths]
        elif _os.path.isfile(seq):
            return _Image.open(seq)
        else:
            raise AssertionError
    elif isinstance(seq,(tuple,list)):
        return [_Image.open(f) for f in seq]

