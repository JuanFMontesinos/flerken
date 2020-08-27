#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as _torch


class _BaseTransformation(object):
    def get_params(self):
        pass

    def reset_params(self):
        pass


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, dim=None):
        self.transforms = transforms
        self.dim = dim

    def __call__(self, inpt):
        if isinstance(inpt, (list, tuple)):
            return self.apply_sequence(inpt)
        else:
            return self.apply_img(inpt)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        output = list(map(self.apply_img, seq))
        if self.dim is not None:
            assert isinstance(self.dim, int)
            output = _torch.stack(output, dim=self.dim)
        for t in self.transforms:
            t.reset_params()
        return output

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


from . import numpy_transforms
from .transforms import *
