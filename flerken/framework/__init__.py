#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import wraps as _wraps

__all__ = ['inference','test','val','train','ctx_iter']


class inference(object):
    def __new__(cls, obj):
        obj.state = cls.__name__
        return obj

    def __call__(self, func):
        @_wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class test(object):
    def __new__(cls, obj):
        assert obj.loaded_model
        obj.state = cls.__name__
        return obj

    def __call__(self, func):
        @_wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class val(object):
    def __new__(cls, obj):

        assert obj.loaded_model
        obj.state = cls.__name__
        return obj

    def __call__(self, func):
        @_wraps(func)
        def decorate_no_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_no_grad


class train(object):
    def __new__(cls, obj):
        assert obj.loaded_model
        obj.state = cls.__name__
        return obj

    def __call__(self, func):
        @_wraps(func)
        def decorate_enable_grad(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_enable_grad

class ctx_iter(object):
    def __new__(cls, obj):
        obj.iterating = True
        return obj

