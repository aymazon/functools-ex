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
    """
    >>> from collections import namedtuple
    >>> from fn import F, _ as X
    >>> from toolz import juxt
    >>> Attr = namedtuple("Attr", ("name", "value"))
    >>> parse_name = F(X.call("partition", "=")[0])
    >>> parse_value = F(X.call("partition", "=")[2])
    >>> load = F(juxt(parse_name, parse_value)) >> F(uppack_args, Attr)
    >>> load("a=b")
    Attr(name=u'a', value=u'b')
    """
    return func(*args)
