#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import wraps as _wraps

__all__ = ['inference', 'test', 'val', 'train', 'ctx_iter']


class Base(object):
    def __new__(cls, obj, state=None):
        if state is not None:
            state = cls.__name__ + '_' + state
        else:
            state = cls.__name__
        obj.state = state
        return obj

    def __call__(self, func):
        @_wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class inference(Base):
    pass


class test(Base):
    pass


class val(Base):
    pass


class train(Base):
    pass


class ctx_iter(object):
    def __new__(cls, obj):
        obj.iterating = True
        return obj
