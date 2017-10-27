# -*- coding: utf-8 -*
""" 补充扩展 fn 和 toolz 中没有的函数库 """

from __future__ import absolute_import, division, print_function, unicode_literals  # NOQA
from functools import partial
from fn.monad import Full, Empty
from toolz import complement, flip

__all__ = ("is_none", "is_not_none", "is_option_full", "is_option_empty",
           "uppack_args")


def is_none(obj):
    return obj is None


is_not_none = complement(is_none)

is_option_full = partial(flip(isinstance), Full)
is_option_empty = partial(flip(isinstance), Empty)


def uppack_args(func, args):
    return func(*args)
